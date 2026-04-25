"""
Task graders for the Loan Approval Environment.

Three difficulty levels:
  EasyGrader   — Collateral-only check (mortgage vs loan).
  MediumGrader — Risk-score threshold check.
  HardGrader   — Full financial × behavioral risk matrix.

Each grader returns a RewardOutput with score, reason,
and risk_breakdown.
"""

from env.action import LoanAction
from env.observation import LoanObservation
from env.reward import RewardOutput


def _get_fraud_score(obs: LoanObservation) -> int:
    """Calculate raw fraud score from behavioral signals."""
    score = 0
    if obs.spending_volatility > 0.7 and obs.loan_amount > 3 * obs.income:
        score += 1
    if obs.employment_years <= 1 and obs.loan_amount > 200000:
        score += 1
    if obs.income < 40000 and obs.loan_amount > 300000:
        score += 1
    return score


class EasyGrader:
    """
    Easy difficulty: pure collateral check.
    APPROVE if mortgage >= 1.1 × loan_amount, else REJECT.
    """

    def grade(self, obs: LoanObservation, action: LoanAction) -> RewardOutput:
        expected = (
            LoanAction.APPROVE
            if obs.mortgage_value >= 1.1 * obs.loan_amount
            else LoanAction.REJECT
        )
        score = 1.0 if action == expected else 0.0
        return RewardOutput(
            score=score,
            reason=(
                f"Expected {expected.value}. "
                f"mortgage={obs.mortgage_value:.0f} "
                f"vs 1.1x loan={1.1 * obs.loan_amount:.0f}"
            ),
            risk_breakdown={"rule": "collateral_check"},
        )


class MediumGrader:
    """
    Medium difficulty: composite risk score threshold.
    APPROVE if risk_score > 0.55, else REJECT.
    REQUEST_MORE_INFO gets partial credit.
    """

    def grade(self, obs: LoanObservation, action: LoanAction) -> RewardOutput:
        rs = (
            0.4 * (obs.credit_score - 300) / 550
            + 0.3 * (obs.income - 30000) / 170000
            + 0.2 * obs.employment_years / 40
            + 0.1 * min(obs.mortgage_value / max(obs.loan_amount, 1), 2) / 2
        )
        expected = LoanAction.APPROVE if rs > 0.55 else LoanAction.REJECT

        if action == expected:
            score = 1.0
        elif action == LoanAction.REQUEST_MORE_INFO:
            score = 0.5
        else:
            score = 0.0

        return RewardOutput(
            score=score,
            reason=f"risk_score={rs:.3f} threshold=0.55",
            risk_breakdown={"risk_score": str(rs)},
        )


class HardGrader:
    """
    Hard difficulty: full financial × behavioral risk matrix.
    Penalizes approving strong fraud or high financial risk.
    """

    def grade(
        self,
        obs: LoanObservation,
        action: LoanAction,
        financial_risk: str,
        behavioral_risk: str,
    ) -> RewardOutput:
        strong_fraud = (
            _get_fraud_score(obs) >= 2 or obs.application_consistency < 0.3
        )

        if strong_fraud and action == LoanAction.APPROVE:
            return RewardOutput(
                score=-1.0,
                reason="Strong fraud signal + APPROVE",
                risk_breakdown={
                    "financial": financial_risk,
                    "behavioral": "STRONG_FRAUD",
                },
            )

        matrix = {
            ("HIGH", "LOW"): LoanAction.REJECT,
            ("HIGH", "MODERATE"): LoanAction.REJECT,
            ("HIGH", "HIGH"): LoanAction.REJECT,
            ("MODERATE", "LOW"): LoanAction.APPROVE,
            ("MODERATE", "MODERATE"): LoanAction.REQUEST_MORE_INFO,
            ("MODERATE", "HIGH"): LoanAction.REJECT,
            ("LOW", "LOW"): LoanAction.APPROVE,
            ("LOW", "MODERATE"): LoanAction.APPROVE,
            ("LOW", "HIGH"): LoanAction.REJECT,
        }
        expected = matrix.get(
            (financial_risk, behavioral_risk), LoanAction.REJECT
        )

        if action == expected:
            score = 1.0
        elif financial_risk == "HIGH" and action == LoanAction.APPROVE:
            score = -1.0
        elif action == LoanAction.REJECT and financial_risk != "LOW":
            score = 0.5
        elif (
            action == LoanAction.REQUEST_MORE_INFO
            and financial_risk == "MODERATE"
            and behavioral_risk == "MODERATE"
        ):
            score = 0.75
        else:
            score = 0.0

        return RewardOutput(
            score=score,
            reason=(
                f"financial={financial_risk} "
                f"behavioral={behavioral_risk} "
                f"expected={expected.value} got={action.value}"
            ),
            risk_breakdown={
                "financial": financial_risk,
                "behavioral": behavioral_risk,
            },
        )
