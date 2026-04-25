"""
R3: Debt Analyzer Sub-Agent.

Rule-based. Zero API calls.
Evaluates: existing_debt, loan_amount, income, loan_term.
"""

from typing import List

from agents.sub_agents.risk.credit_analyst import SubAgentReport
from env.observation import LoanObservation


class DebtAnalyzerAgent:
    """
    Analyzes debt-to-income ratio and EMI burden
    to determine if the applicant can service the loan.
    """

    def analyze(self, obs: LoanObservation) -> SubAgentReport:
        # Debt-to-income ratio
        dti = obs.existing_debt / max(obs.income, 1)
        # EMI as fraction of monthly income
        monthly_emi = obs.loan_amount / (obs.loan_term * 12)
        monthly_income = obs.income / 12
        emi = monthly_emi / max(monthly_income, 1)

        flags: List[str] = [f"dti_{dti:.2f}", f"emi_burden_{emi:.2f}"]

        # Determine recommendation
        if dti < 0.3 and emi < 0.4:
            recommendation = "APPROVE"
            confidence = 0.8
            reasoning = (
                f"Debt levels are manageable. DTI={dti:.2f} and "
                f"EMI burden={emi:.2f} are both within safe limits. "
                f"Existing debt {obs.existing_debt:.0f} against income {obs.income:.0f}."
            )
        elif dti < 0.5 and emi < 0.5:
            recommendation = "REVIEW"
            confidence = 0.6
            reasoning = (
                f"Moderate debt exposure. DTI={dti:.2f}, EMI burden={emi:.2f}. "
                f"Existing debt {obs.existing_debt:.0f} with loan request {obs.loan_amount:.0f} "
                f"over {obs.loan_term} years needs review."
            )
        else:
            recommendation = "REJECT"
            confidence = 0.85
            reasoning = (
                f"High debt risk. DTI={dti:.2f} and/or EMI burden={emi:.2f} "
                f"exceed safe thresholds. Total debt obligation would be "
                f"{obs.existing_debt + obs.loan_amount:.0f}."
            )

        return SubAgentReport(
            agent_name="DebtAnalyzer",
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            key_flags=flags,
        )
