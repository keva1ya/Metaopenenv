"""
Reward output model for the grading system.

Used by all three grader difficulty levels to return
structured reward information.
"""

from typing import Dict

from pydantic import BaseModel


class RewardOutput(BaseModel):
    """
    Structured reward output from a grader.

    Attributes:
        score: Reward value, typically in [0.0, 1.0].
        reason: Human-readable explanation of the reward.
        risk_breakdown: Dictionary of risk factors evaluated.
    """

    score: float
    reason: str
    risk_breakdown: Dict[str, str]
