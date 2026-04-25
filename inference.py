"""
Inference script for the Loan Approval OpenEnv system.

Runs the active decision flow across all three task levels
(easy, medium, hard) with 5 episodes each.

Hard mode uses the weighted decision engine.
Calculates a normalized final score.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain.dynamic_weight_net import DynamicWeightDecisionEngine
from env.action import LoanAction
from env.loan_env import LoanApprovalEnv


def run_inference() -> None:
    """
    Run inference across all task levels.
    Easy: collateral-based heuristic.
    Medium: risk-score-based heuristic.
    Hard: weighted decision engine.
    """
    decision_engine = DynamicWeightDecisionEngine(feature_dim=28)

    all_rewards = []

    for task in ["easy", "medium", "hard"]:
        env = LoanApprovalEnv(task_level=task, seed=42)
        task_rewards = []

        print(f"\n{'=' * 60}")
        print(f"Task: {task.upper()}")
        print(f"{'=' * 60}")

        for ep in range(5):
            obs = env.reset()

            if task == "hard":
                council = env.get_council_recommendation()
                result = decision_engine.decide(
                    obs=obs,
                    council=council,
                    portfolio=env.portfolio,
                    manual_multipliers={
                        "RiskAgent": 33.0,
                        "FraudAgent": 33.0,
                        "PolicyAgent": 34.0,
                    },
                )
                action = result.action
            elif task == "medium":
                rs = (
                    0.4 * (obs.credit_score - 300) / 550
                    + 0.3 * (obs.income - 30000) / 170000
                    + 0.2 * obs.employment_years / 40
                    + 0.1
                    * min(
                        obs.mortgage_value / max(obs.loan_amount, 1),
                        2,
                    )
                    / 2
                )
                action = (
                    LoanAction.APPROVE if rs > 0.55 else LoanAction.REJECT
                )
            else:
                action = (
                    LoanAction.APPROVE
                    if obs.mortgage_value >= 1.1 * obs.loan_amount
                    else LoanAction.REJECT
                )

            if task == "hard":
                _, reward, _, _ = env.step(action, council_result=council)
            else:
                _, reward, _, _ = env.step(action)
            task_rewards.append(reward)

            print(
                f"  Episode {ep + 1}: "
                f"action={action.value} "
                f"reward={reward:.4f}"
            )

        avg = sum(task_rewards) / len(task_rewards)
        all_rewards.extend(task_rewards)
        print(f"  Average: {avg:.4f}")

    total = sum(all_rewards)
    score = max(0.01, min(0.99, 0.01 + (total / 15.0) * 0.98))

    print(f"\n{'=' * 60}")
    print(f"TOTAL REWARD: {total:.4f}")
    print(f"FINAL SCORE: {score:.6f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_inference()
