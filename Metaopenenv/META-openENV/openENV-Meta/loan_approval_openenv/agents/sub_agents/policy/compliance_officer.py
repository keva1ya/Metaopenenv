"""
P1: Compliance Officer Sub-Agent.

Rule-based. Zero API calls.
Evaluates: age, loan_term, customer_score.
Enforces bank policy age requirements and retirement limits.
"""

from typing import List

from agents.sub_agents.risk.credit_analyst import SubAgentReport
from env.observation import LoanObservation


class ComplianceOfficerAgent:
    """
    Checks regulatory and bank policy compliance
    related to applicant age and loan term constraints.
    """

    def analyze(self, obs: LoanObservation) -> SubAgentReport:
        flags: List[str] = []

        # Hard rule: age limits
        if obs.age < 21 or obs.age > 60:
            flags.append("age_policy_violation")
            return SubAgentReport(
                agent_name="ComplianceOfficer",
                recommendation="REJECT",
                confidence=1.0,
                reasoning=(
                    f"Age {obs.age} violates bank policy (must be 21-60). "
                    f"This is a hard compliance rejection. "
                    f"No exceptions allowed."
                ),
                key_flags=flags,
            )

        # Hard rule: age + term exceeds retirement
        if obs.age + obs.loan_term > 70:
            flags.append("exceeds_retirement_age")
            return SubAgentReport(
                agent_name="ComplianceOfficer",
                recommendation="REJECT",
                confidence=1.0,
                reasoning=(
                    f"Age {obs.age} + term {obs.loan_term} = {obs.age + obs.loan_term} "
                    f"exceeds retirement limit of 70. "
                    f"Loan would extend past safe repayment window."
                ),
                key_flags=flags,
            )

        # Loyalty bonus check
        if obs.customer_score >= 0.8:
            flags.append("loyalty_bonus_eligible")
            return SubAgentReport(
                agent_name="ComplianceOfficer",
                recommendation="APPROVE",
                confidence=0.9,
                reasoning=(
                    f"Applicant age {obs.age} is compliant. "
                    f"Customer score {obs.customer_score:.2f} qualifies for loyalty bonus. "
                    f"Term {obs.loan_term}yr within retirement window."
                ),
                key_flags=flags,
            )

        # Default: compliant
        return SubAgentReport(
            agent_name="ComplianceOfficer",
            recommendation="APPROVE",
            confidence=0.8,
            reasoning=(
                f"Age {obs.age} and term {obs.loan_term}yr comply with policy. "
                f"Age + term = {obs.age + obs.loan_term} is within 70yr limit. "
                f"Standard approval without loyalty bonus."
            ),
            key_flags=flags,
        )
