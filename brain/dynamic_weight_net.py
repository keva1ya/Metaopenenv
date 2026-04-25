"""
Dynamic weighting engine for final loan decisions.

This module replaces the BigBrain DQN decision path with a
Mixture-of-Experts style combiner:
  - Risk head agent
  - Fraud head agent
  - Policy head agent

The final action is selected from weighted action logits.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from agents.voting import VoteResult
from brain.state_builder import PortfolioState, build_state_vector
from env.action import LoanAction
from env.observation import LoanObservation


@dataclass
class WeightedDecisionResult:
    """Final weighted decision payload."""

    action: LoanAction
    normalized_weights: Dict[str, float]
    weighted_action_scores: Dict[str, float]
    uncertainty_score: float


class DynamicWeightNet(nn.Module):
    """
    Learnable weight network.
    Input: applicant feature vector + 3 confidence values.
    Output: softmax weights [risk, fraud, policy].
    """

    def __init__(self, feature_dim: int = 28, n_agents: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim + n_agents, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_agents),
            nn.Softmax(dim=-1),
        )

    def forward(
        self, features: torch.Tensor, confidences: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([features, confidences], dim=-1)
        return self.net(x)


class DynamicWeightDecisionEngine:
    """
    Runtime decision engine using weighted head-agent votes.

    For now, deterministic confidence-weighted aggregation is used by default.
    The neural weight net is kept as an optional path for future end-to-end
    training once labeled data integration is enabled.
    """

    ACTION_INDEX = {
        LoanAction.APPROVE: 0,
        LoanAction.REJECT: 1,
        LoanAction.REQUEST_MORE_INFO: 2,
    }

    INDEX_ACTION = {v: k for k, v in ACTION_INDEX.items()}

    AGENT_ORDER = ("RiskAgent", "FraudAgent", "PolicyAgent")

    def __init__(
        self,
        feature_dim: int = 28,
        use_neural_weights: bool = False,
    ):
        self.weight_net = DynamicWeightNet(feature_dim=feature_dim, n_agents=3)
        self.use_neural_weights = use_neural_weights

    def decide(
        self,
        obs: LoanObservation,
        council: VoteResult,
        portfolio: PortfolioState,
        manual_multipliers: Dict[str, float],
    ) -> WeightedDecisionResult:
        """Compute weighted final decision from head-agent outputs."""
        state_vector = build_state_vector(obs, council, portfolio)
        confidence_vec = np.array(
            [
                float(council.agent_confidence.get("RiskAgent", 0.0)),
                float(council.agent_confidence.get("FraudAgent", 0.0)),
                float(council.agent_confidence.get("PolicyAgent", 0.0)),
            ],
            dtype=np.float32,
        )

        if self.use_neural_weights:
            features = torch.FloatTensor(state_vector).unsqueeze(0)
            conf = torch.FloatTensor(confidence_vec).unsqueeze(0)
            with torch.no_grad():
                raw_weights = self.weight_net(features, conf).squeeze(0).numpy()
        else:
            raw_weights = self._manual_weight_mix(council, manual_multipliers)

        normalized_weights = self._normalize(raw_weights)
        score_vec = np.zeros(3, dtype=np.float32)

        for idx, agent_name in enumerate(self.AGENT_ORDER):
            vote_str = council.vote_breakdown.get(agent_name, LoanAction.REJECT.value)
            try:
                action = LoanAction(vote_str)
            except ValueError:
                action = LoanAction.REJECT
            score_vec[self.ACTION_INDEX[action]] += normalized_weights[idx]

        winner_idx = int(np.argmax(score_vec))
        action = self.INDEX_ACTION[winner_idx]
        uncertainty_score = float(1.0 - np.max(score_vec))

        return WeightedDecisionResult(
            action=action,
            normalized_weights={
                "RiskAgent": float(normalized_weights[0]),
                "FraudAgent": float(normalized_weights[1]),
                "PolicyAgent": float(normalized_weights[2]),
            },
            weighted_action_scores={
                LoanAction.APPROVE.value: float(score_vec[0]),
                LoanAction.REJECT.value: float(score_vec[1]),
                LoanAction.REQUEST_MORE_INFO.value: float(score_vec[2]),
            },
            uncertainty_score=uncertainty_score,
        )

    def _manual_weight_mix(
        self,
        council: VoteResult,
        manual_multipliers: Dict[str, float],
    ) -> np.ndarray:
        conf = np.array(
            [
                max(float(council.agent_confidence.get("RiskAgent", 0.0)), 0.0),
                max(float(council.agent_confidence.get("FraudAgent", 0.0)), 0.0),
                max(float(council.agent_confidence.get("PolicyAgent", 0.0)), 0.0),
            ],
            dtype=np.float32,
        )
        multipliers = np.array(
            [
                max(float(manual_multipliers.get("RiskAgent", 1.0)), 0.0),
                max(float(manual_multipliers.get("FraudAgent", 1.0)), 0.0),
                max(float(manual_multipliers.get("PolicyAgent", 1.0)), 0.0),
            ],
            dtype=np.float32,
        )
        return conf * multipliers

    @staticmethod
    def _normalize(weights: np.ndarray) -> np.ndarray:
        total = float(weights.sum())
        if total <= 1e-8:
            return np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        return weights / total

