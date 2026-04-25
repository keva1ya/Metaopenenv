"""
P2: Term Validator Sub-Agent.

Rule-based. Zero API calls.
Evaluates: loan_term, loan_amount, income.
Validates loan term reasonableness and EMI affordability.
"""

from typing import List

from agents.sub_agents.risk.credit_analyst import SubAgentReport
from env.observation import LoanObservation


class TermValidatorAgent:
    """
    Validates that the loan term is reasonable given
    the loan amount and applicant income.
    """

    def analyze(self, obs: LoanObservation) -> SubAgentReport:
        flags: List[str] = []

        # Hard rule: term exceeds 25-year limit
        if obs.loan_term > 25:
            flags.append("term_exceeds_25yr_limit")
            return SubAgentReport(
                agent_name="TermValidator",
                recommendation="REJECT",
                confidence=1.0,
                reasoning=(
                    f"Loan term {obs.loan_term}yr exceeds maximum 25-year limit. "
                    f"This is a hard policy constraint. "
                    f"Applicant must reduce term."
                ),
                key_flags=flags,
            )

        monthly_install = obs.loan_amount / (obs.loan_term * 12)
        monthly_income = obs.income / 12

        # Short term with large amount
        if obs.loan_term < 5 and obs.loan_amount > 300000:
            flags.append("short_term_large_amount")
            return SubAgentReport(
                agent_name="TermValidator",
                recommendation="REVIEW",
                confidence=0.7,
                reasoning=(
                    f"Short term {obs.loan_term}yr with large loan {obs.loan_amount:.0f} "
                    f"creates high monthly payments ({monthly_install:.0f}/month). "
                    f"Needs review for payment capacity."
                ),
                key_flags=flags,
            )

        # EMI exceeds 50% of monthly income
        if monthly_install > 0.5 * monthly_income:
            flags.append("emi_gt_50pct_income")
            return SubAgentReport(
                agent_name="TermValidator",
                recommendation="REVIEW",
                confidence=0.65,
                reasoning=(
                    f"Monthly installment {monthly_install:.0f} exceeds 50% of "
                    f"monthly income {monthly_income:.0f}. "
                    f"EMI is {monthly_install / max(monthly_income, 1) * 100:.1f}% of income."
                ),
                key_flags=flags,
            )

        # All checks pass
        flags.append(f"emi_pct_{monthly_install / max(monthly_income, 1) * 100:.1f}")
        return SubAgentReport(
            agent_name="TermValidator",
            recommendation="APPROVE",
            confidence=0.85,
            reasoning=(
                f"Loan term {obs.loan_term}yr is reasonable. "
                f"Monthly EMI {monthly_install:.0f} is "
                f"{monthly_install / max(monthly_income, 1) * 100:.1f}% of income. "
                f"Within safe limits."
            ),
            key_flags=flags,
        )
