"""
R2: Income Verifier Sub-Agent.

Rule-based. Zero API calls.
Evaluates: income, employment_years, monthly_expenses,
           num_dependents, gst_registered.
"""

from typing import List

from agents.sub_agents.risk.credit_analyst import SubAgentReport
from env.observation import LoanObservation


class IncomeVerifierAgent:
    """
    Verifies income adequacy by computing disposable income
    ratio after expenses and dependent costs.
    """

    def analyze(self, obs: LoanObservation) -> SubAgentReport:
        flags: List[str] = []

        # Disposable income calculation
        annual_expenses = obs.monthly_expenses * 12
        dependent_cost = obs.num_dependents * 30000
        disposable = obs.income - annual_expenses - dependent_cost
        disposable_ratio = disposable / max(obs.income, 1)

        flags.append(f"disposable_ratio_{disposable_ratio:.2f}")
        flags.append(f"employment_{obs.employment_years}yr")

        if obs.gst_registered:
            flags.append("gst_positive_signal")

        # Determine recommendation
        if disposable_ratio >= 0.2 and obs.employment_years >= 2:
            recommendation = "APPROVE"
            confidence = 0.8
            reasoning = (
                f"Income is adequate with disposable ratio {disposable_ratio:.2f} "
                f"and {obs.employment_years} years employment. "
                f"Net disposable income: {disposable:.0f}."
            )
        elif disposable_ratio >= 0.1:
            recommendation = "REVIEW"
            confidence = 0.55
            reasoning = (
                f"Disposable ratio {disposable_ratio:.2f} is marginal. "
                f"Income {obs.income:.0f} minus expenses {annual_expenses:.0f} "
                f"and {obs.num_dependents} dependents leaves limited buffer."
            )
        else:
            recommendation = "REJECT"
            confidence = 0.85
            reasoning = (
                f"Insufficient disposable income (ratio {disposable_ratio:.2f}). "
                f"After annual expenses {annual_expenses:.0f} and "
                f"{obs.num_dependents} dependents, net is {disposable:.0f}."
            )

        return SubAgentReport(
            agent_name="IncomeVerifier",
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            key_flags=flags,
        )
