"""
State Builder for Big Brain DQN.

Converts the raw observation, council vote result, and
portfolio health into a normalized 28-dimensional float32
state vector for the DQN.

Dimensions:
  [0-13]  14 financial features (normalized)
  [14-16]  3 behavioral features
  [17-19]  3 council vote encodings
  [20]     1 confidence average
  [21]     1 unanimity flag
  [22]     1 fraud score signal
  [23-26]  4 portfolio health features
  [27]     1 dissent signal

Total: 28
"""

from collections import deque
from typing import Optional

import numpy as np

from agents.voting import VoteResult
from env.observation import LoanObservation


class PortfolioState:
    """
    Tracks portfolio-level health across episodes.
    Persists within one env instance.
    Resets only when env is re-instantiated.
    """

    def __init__(self):
        self.total_approved: int = 0
        self.total_rejected: int = 0
        self.total_reviewed: int = 0
        self.total_defaults: int = 0
        self.current_capital: float = 1_000_000.0
        self.reward_window: deque = deque(maxlen=10)

    @property
    def avg_reward_last_10(self) -> float:
        """Average reward over the last 10 episodes."""
        if not self.reward_window:
            return 0.0
        return float(np.mean(list(self.reward_window)))

    def update(self, action: str, reward: float) -> None:
        """Update portfolio stats after a decision."""
        if action == "APPROVE":
            self.total_approved += 1
        elif action == "REJECT":
            self.total_rejected += 1
        else:
            self.total_reviewed += 1
        self.reward_window.append(reward)
        if reward == -1.0:
            self.total_defaults += 1


def build_state_vector(
    obs: LoanObservation,
    council_result: VoteResult,
    portfolio: PortfolioState,
) -> np.ndarray:
    """
    Build a 28-dimensional normalized state vector
    for the Big Brain DQN.

    All values are normalized to approximately [0, 1].
    """
    ACTION_ENC = {
        "APPROVE": 1.0,
        "REQUEST_MORE_INFO": 0.5,
        "REJECT": 0.0,
    }

    # ── Financial features (14 values) ───────────────────
    financial = [
        (obs.income - 30000) / 170000,
        (obs.credit_score - 300) / 550,
        obs.employment_years / 40,
        (obs.loan_amount - 50000) / 450000,
        obs.mortgage_value / 700000,
        obs.loan_term / 30,
        obs.existing_debt / 200000,
        (obs.age - 18) / 47,
        (obs.monthly_expenses - 5000) / 75000,
        obs.num_dependents / 8,
        obs.previous_defaults / 5,
        float(obs.gst_registered),
        obs.itr_filed_years / 20,
        obs.customer_score,
    ]

    # ── Behavioral features (3 values) ───────────────────
    behavioral = [
        obs.spending_volatility,
        obs.loan_purpose_clarity,
        obs.application_consistency,
    ]

    # ── Council vote encodings (3 values) ────────────────
    council_votes = [
        ACTION_ENC.get(
            council_result.vote_breakdown.get("RiskAgent", "REJECT"),
            0.0,
        ),
        ACTION_ENC.get(
            council_result.vote_breakdown.get("FraudAgent", "REJECT"),
            0.0,
        ),
        ACTION_ENC.get(
            council_result.vote_breakdown.get("PolicyAgent", "REJECT"),
            0.0,
        ),
    ]

    # ── Confidence average (1 value) ─────────────────────
    conf = [council_result.confidence_average]

    # ── Unanimity flag (1 value) ─────────────────────────
    unanimous = [float(council_result.was_unanimous)]

    # ── Fraud score signal (1 value) ─────────────────────
    fraud_score_val = 0.0
    for r in council_result.all_sub_reports:
        if r.agent_name == "PatternChecker":
            for flag in r.key_flags:
                if "fraud_score:" in flag:
                    try:
                        fraud_score_val = (
                            float(
                                flag.split(":")[1]
                                .split("/")[0]
                                .strip()
                            )
                            / 3.0
                        )
                    except (ValueError, IndexError):
                        pass
    fraud_signal = [fraud_score_val]

    # ── Portfolio health features (4 values) ─────────────
    portfolio_features = [
        min(portfolio.total_approved / 100, 1.0),
        min(
            portfolio.total_defaults / max(portfolio.total_approved, 1),
            1.0,
        ),
        min(portfolio.current_capital / 1_000_000, 1.0),
        float(np.clip(portfolio.avg_reward_last_10, -1, 1)) * 0.5
        + 0.5,
    ]

    # ── Dissent signal (1 value) ─────────────────────────
    dissent_signal = [
        1.0 if council_result.dissent_log else 0.0
    ]

    # TOTAL: 14 + 3 + 3 + 1 + 1 + 1 + 4 + 1 = 28
    state = (
        financial
        + behavioral
        + council_votes
        + conf
        + unanimous
        + fraud_signal
        + portfolio_features
        + dissent_signal
    )

    return np.array(state, dtype=np.float32)
