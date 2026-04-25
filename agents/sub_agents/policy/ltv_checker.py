"""
P3: LTV Checker Sub-Agent.

Rule-based. Zero API calls.
Evaluates: mortgage_value, loan_amount, existing_debt.
Checks Loan-to-Value ratio for secured loans.
"""

from typing import List

from agents.sub_agents.risk.credit_analyst import SubAgentReport
from env.observation import LoanObservation


class LTVCheckerAgent:
    """
    Evaluates Loan-to-Value (LTV) ratio to assess
    collateral adequacy for the requested loan.
    """

    def analyze(self, obs: LoanObservation) -> SubAgentReport:
        flags: List[str] = []

        # Unsecured loan (no collateral)
        if obs.mortgage_value == 0:
            flags.append("unsecured_no_collateral")
            return SubAgentReport(
                agent_name="LTVChecker",
                recommendation="REVIEW",
                confidence=0.7,
                reasoning=(
                    f"No collateral provided (mortgage_value=0). "
                    f"Loan of {obs.loan_amount:.0f} is unsecured. "
                    f"Higher risk requires additional review."
                ),
                key_flags=flags,
            )

        # Calculate LTV
        ltv = obs.loan_amount / obs.mortgage_value

        if ltv > 0.9:
            flags.append(f"ltv_{ltv:.2f}")
            return SubAgentReport(
                agent_name="LTVChecker",
                recommendation="REJECT",
                confidence=0.85,
                reasoning=(
                    f"LTV ratio {ltv:.2f} exceeds 90% threshold. "
                    f"Loan {obs.loan_amount:.0f} vs collateral {obs.mortgage_value:.0f}. "
                    f"Insufficient collateral coverage."
                ),
                key_flags=flags,
            )
        elif ltv > 0.7:
            flags.append(f"ltv_{ltv:.2f}")
            return SubAgentReport(
                agent_name="LTVChecker",
                recommendation="REVIEW",
                confidence=0.65,
                reasoning=(
                    f"LTV ratio {ltv:.2f} is between 70-90%. "
                    f"Collateral {obs.mortgage_value:.0f} provides moderate coverage "
                    f"for loan {obs.loan_amount:.0f}."
                ),
                key_flags=flags,
            )
        else:
            flags.append(f"ltv_{ltv:.2f}")
            return SubAgentReport(
                agent_name="LTVChecker",
                recommendation="APPROVE",
                confidence=0.9,
                reasoning=(
                    f"LTV ratio {ltv:.2f} is below 70% — strong collateral. "
                    f"Collateral {obs.mortgage_value:.0f} well exceeds "
                    f"loan {obs.loan_amount:.0f}."
                ),
                key_flags=flags,
            )
