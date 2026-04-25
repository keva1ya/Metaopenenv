"""
R1: Credit Analyst Sub-Agent.

Rule-based. Zero API calls.
Evaluates: credit_score, previous_defaults,
           customer_score, itr_filed_years.
"""

from typing import List

from pydantic import BaseModel

from env.observation import LoanObservation


class SubAgentReport(BaseModel):
    """Standardized output from any sub-agent."""

    agent_name: str
    recommendation: str   # "APPROVE" / "REJECT" / "REVIEW"
    confidence: float     # 0.0 to 1.0
    reasoning: str        # 2-3 sentences
    key_flags: List[str]  # specific concerns or positives


class CreditAnalystAgent:
    """
    Analyzes credit-worthiness based on credit score,
    default history, ITR filing, and customer loyalty.
    """

    def analyze(self, obs: LoanObservation) -> SubAgentReport:
        flags: List[str] = []

        # Calculate composite risk
        risk = 0.0
        if obs.credit_score > 650:
            risk = 0.2
        elif obs.credit_score > 550:
            risk = 0.5
        else:
            risk = 0.8
            flags.append(f"low_credit_score_{obs.credit_score}")

        risk += obs.previous_defaults * 0.1
        if obs.previous_defaults > 0:
            flags.append(f"defaults_{obs.previous_defaults}")

        if obs.itr_filed_years < 2:
            risk += 0.1
            flags.append(f"low_itr_years_{obs.itr_filed_years}")

        if obs.customer_score > 0.7:
            risk -= 0.1
            flags.append("high_customer_loyalty")

        risk = max(0.0, min(1.0, risk))

        # Determine recommendation
        if risk < 0.3:
            recommendation = "APPROVE"
            confidence = 1.0 - risk
            reasoning = (
                f"Credit profile is strong with score {obs.credit_score} "
                f"and {obs.previous_defaults} defaults. "
                f"Composite risk {risk:.2f} is below threshold."
            )
        elif risk < 0.6:
            recommendation = "REVIEW"
            confidence = 0.6
            reasoning = (
                f"Credit profile shows moderate risk ({risk:.2f}). "
                f"Score {obs.credit_score} with {obs.previous_defaults} defaults "
                f"warrants further review."
            )
        else:
            recommendation = "REJECT"
            confidence = risk
            reasoning = (
                f"Credit profile is high-risk ({risk:.2f}). "
                f"Score {obs.credit_score}, {obs.previous_defaults} defaults, "
                f"and {obs.itr_filed_years} ITR years are concerning."
            )

        flags.append(f"credit_risk_{risk:.2f}")

        return SubAgentReport(
            agent_name="CreditAnalyst",
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            key_flags=flags,
        )
