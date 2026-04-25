"""
F1: Pattern Checker Sub-Agent.

Rule-based. Zero API calls.
Evaluates: spending_volatility, loan_amount, income,
           employment_years.
Detects fraud patterns via heuristic scoring.
"""

from typing import List

from agents.sub_agents.risk.credit_analyst import SubAgentReport
from env.observation import LoanObservation


class PatternCheckerAgent:
    """
    Detects fraud patterns by checking anomalous combinations
    of spending volatility, loan size, income, and employment.
    """

    def analyze(self, obs: LoanObservation) -> SubAgentReport:
        fraud_score = 0
        flags: List[str] = []

        # Pattern 1: High volatility + disproportionate loan
        if obs.spending_volatility > 0.7 and obs.loan_amount > 3 * obs.income:
            fraud_score += 1
            flags.append("high_volatility_high_loan")

        # Pattern 2: New employee requesting large loan
        if obs.employment_years <= 1 and obs.loan_amount > 200000:
            fraud_score += 1
            flags.append("new_employee_large_loan")

        # Pattern 3: Low income requesting very large loan
        if obs.income < 40000 and obs.loan_amount > 300000:
            fraud_score += 1
            flags.append("income_loan_mismatch")

        flags.append(f"fraud_score:{fraud_score}/3")

        # Determine recommendation
        if fraud_score >= 2:
            recommendation = "REJECT"
            confidence = 0.9
            reasoning = (
                f"Multiple fraud patterns detected (score {fraud_score}/3). "
                f"Spending volatility {obs.spending_volatility:.2f} with "
                f"loan {obs.loan_amount:.0f} vs income {obs.income:.0f} "
                f"raises significant red flags."
            )
        elif fraud_score == 1:
            recommendation = "REVIEW"
            confidence = 0.65
            reasoning = (
                f"One fraud pattern detected (score {fraud_score}/3). "
                f"Requires additional scrutiny. "
                f"Volatility={obs.spending_volatility:.2f}, employment={obs.employment_years}yr."
            )
        else:
            recommendation = "APPROVE"
            confidence = 0.8
            reasoning = (
                f"No fraud patterns detected (score 0/3). "
                f"Spending patterns and loan request appear consistent "
                f"with income {obs.income:.0f} and employment {obs.employment_years}yr."
            )

        return SubAgentReport(
            agent_name="PatternChecker",
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            key_flags=flags,
        )
