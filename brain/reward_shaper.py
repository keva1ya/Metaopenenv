"""
Reward Shaper for Big Brain training.

Four independent reward components combined:
  r1: Financial reward   (weight 0.4)
  r2: Fraud reward       (weight 0.3)
  r3: Policy reward      (weight 0.2)
  r4: Alignment reward   (weight 0.1)

Hard overrides checked first to prevent reward hacking.
Multiple signals prevent the agent from gaming any single metric.
"""

from typing import Optional

import numpy as np

from agents.voting import VoteResult
from env.observation import LoanObservation


def calculate_shaped_reward(
    obs: LoanObservation,
    action: str,
    council_result: Optional[VoteResult],
    task_level: str,
) -> float:
    """
    Calculate multi-component shaped reward for the Big Brain.

    Hard overrides are checked first and return immediately.
    Then four weighted components are combined and clipped.
    """
    # HARD OVERRIDES (check first, return immediately)
    if _is_strong_fraud(obs) and action == "APPROVE":
        return -1.0
    if _is_high_financial_risk(obs) and action == "APPROVE":
        return -1.0

    # MORTGAGE_MODE shortcut
    if obs.mortgage_value >= 1.1 * obs.loan_amount:
        return 1.0 if action == "APPROVE" else 0.0

    # FOUR REWARD COMPONENTS
    r1 = _financial_reward(obs, action)     # weight 0.4
    r2 = _fraud_reward(obs, action)         # weight 0.3
    r3 = _policy_reward(obs, action)        # weight 0.2
    r4 = _alignment_reward(action, council_result)  # weight 0.1

    total = 0.4 * r1 + 0.3 * r2 + 0.2 * r3 + 0.1 * r4
    return float(np.clip(total, -1.0, 1.0))


def _financial_reward(obs: LoanObservation, action: str) -> float:
    """Reward based on financial risk assessment."""
    if obs.mortgage_value >= 1.1 * obs.loan_amount:
        return 1.0 if action == "APPROVE" else -0.5

    risk_score = (
        0.4 * (obs.credit_score - 300) / 550
        + 0.3 * (obs.income - 30000) / 170000
        + 0.2 * obs.employment_years / 40
        + 0.1
        * min(obs.mortgage_value / max(obs.loan_amount, 1), 2)
        / 2
    )

    if risk_score > 0.55:
        return 1.0 if action == "APPROVE" else 0.0
    else:
        return 1.0 if action == "REJECT" else -0.5


def _fraud_reward(obs: LoanObservation, action: str) -> float:
    """Reward based on fraud signal strength."""
    fs = _get_fraud_score(obs)
    if fs >= 2:
        return 1.0 if action == "REJECT" else -1.0
    elif fs == 1:
        return 0.5 if action == "REQUEST_MORE_INFO" else 0.0
    else:
        return 0.3 if action == "APPROVE" else 0.0


def _policy_reward(obs: LoanObservation, action: str) -> float:
    """Reward based on policy compliance."""
    violations = 0
    if obs.age < 21 or obs.age > 60:
        violations += 1
    if obs.age + obs.loan_term > 70:
        violations += 1
    if obs.loan_term > 25:
        violations += 1

    threshold = 0.6 if obs.customer_score >= 0.8 else 0.5
    if (
        (obs.existing_debt + obs.loan_amount)
        / (obs.income * obs.loan_term)
    ) >= threshold:
        violations += 1

    if obs.loan_amount > 10 * obs.income:
        violations += 1

    if violations > 0:
        return 1.0 if action == "REJECT" else -0.5
    return 1.0 if action == "APPROVE" else -0.3


def _alignment_reward(
    action: str, council_result: Optional[VoteResult]
) -> float:
    """Reward for aligning with unanimous council votes."""
    if council_result is None:
        return 0.0
    if not council_result.was_unanimous:
        return 0.0
    council_action = council_result.council_recommendation.value
    return 0.1 if action == council_action else -0.1


def _get_fraud_score(obs: LoanObservation) -> int:
    """Calculate raw fraud score (0-3) from behavioral signals."""
    score = 0
    if (
        obs.spending_volatility > 0.7
        and obs.loan_amount > 3 * obs.income
    ):
        score += 1
    if (
        obs.employment_years <= 1
        and obs.loan_amount > 200000
    ):
        score += 1
    if obs.income < 40000 and obs.loan_amount > 300000:
        score += 1
    return score


def _is_strong_fraud(obs: LoanObservation) -> bool:
    """Detect strong fraud signal (hard override trigger)."""
    return (
        _get_fraud_score(obs) >= 2
        or obs.application_consistency < 0.3
    )


def _is_high_financial_risk(obs: LoanObservation) -> bool:
    """Detect high financial risk (hard override trigger)."""
    rs = (
        0.4 * (obs.credit_score - 300) / 550
        + 0.3 * (obs.income - 30000) / 170000
        + 0.2 * obs.employment_years / 40
        + 0.1
        * min(obs.mortgage_value / max(obs.loan_amount, 1), 2)
        / 2
    )
    return rs < 0.3
