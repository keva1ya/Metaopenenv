"""
F2: Consistency Checker Sub-Agent.

Rule-based. Zero API calls.
Evaluates: application_consistency, loan_purpose_clarity,
           previous_defaults, spending_volatility.
Checks for misrepresentation and inconsistency patterns.
"""

from typing import List

from agents.sub_agents.risk.credit_analyst import SubAgentReport
from env.observation import LoanObservation


class ConsistencyCheckerAgent:
    """
    Detects inconsistencies in application data that may
    indicate misrepresentation or identity fraud.
    """

    def analyze(self, obs: LoanObservation) -> SubAgentReport:
        flags: List[str] = []

        # Check 1: Low application consistency
        if obs.application_consistency < 0.4:
            flags.append("low_consistency")

        # Check 2: Vague purpose with large loan
        if obs.loan_purpose_clarity < 0.3 and obs.loan_amount > 200000:
            flags.append("vague_purpose_large_loan")

        # Check 3: Misrepresentation pattern
        if obs.previous_defaults > 1 and obs.application_consistency < 0.5:
            flags.append("misrepresentation_pattern")

        # Determine recommendation
        if len(flags) >= 2:
            recommendation = "REJECT"
            confidence = 0.85
            reasoning = (
                f"Multiple consistency issues detected ({len(flags)} flags). "
                f"Application consistency={obs.application_consistency:.2f}, "
                f"purpose clarity={obs.loan_purpose_clarity:.2f}. "
                f"High risk of misrepresentation."
            )
        elif len(flags) == 1:
            recommendation = "REVIEW"
            confidence = 0.6
            reasoning = (
                f"One consistency issue found: {flags[0]}. "
                f"Application consistency={obs.application_consistency:.2f}, "
                f"previous defaults={obs.previous_defaults}."
            )
        else:
            recommendation = "APPROVE"
            confidence = 0.8
            reasoning = (
                f"Application appears consistent. "
                f"Consistency={obs.application_consistency:.2f}, "
                f"purpose clarity={obs.loan_purpose_clarity:.2f}. "
                f"No misrepresentation indicators."
            )

        return SubAgentReport(
            agent_name="ConsistencyChecker",
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            key_flags=flags,
        )
