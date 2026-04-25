"""
Inference script for the Loan Approval OpenEnv system.

Runs the Big Brain DQN agent across all three task levels
(easy, medium, hard) with 5 episodes each.

Loads trained weights if available.
Calculates a normalized final score.
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from brain.big_brain import BigBrainAgent
from brain.state_builder import build_state_vector
from env.action import LoanAction
from env.loan_env import LoanApprovalEnv


def run_inference() -> None:
    """
    Run inference across all task levels.
    Easy: collateral-based heuristic.
    Medium: risk-score-based heuristic.
    Hard: Big Brain DQN agent.
    """
    brain = BigBrainAgent(state_dim=28, epsilon=0.0)
    brain.load("brain/big_brain_weights.pt")

    all_rewards = []

    for task in ["easy", "medium", "hard"]:
        env = LoanApprovalEnv(task_level=task, seed=42)
        task_rewards = []

        print(f"\n{'='*60}")
        print(f"Task: {task.upper()}")
        print(f"{'='*60}")

        for ep in range(5):
            obs = env.reset()

            if task == "hard":
                # Full pipeline: council → Big Brain
                council = env.get_council_recommendation()
                state = build_state_vector(
                    obs, council, env.portfolio
                )
                action_str = brain.decide(state)
                action = LoanAction(action_str)
            elif task == "medium":
                # Risk-score heuristic
                rs = (
                    0.4 * (obs.credit_score - 300) / 550
                    + 0.3 * (obs.income - 30000) / 170000
                    + 0.2 * obs.employment_years / 40
                    + 0.1
                    * min(
                        obs.mortgage_value
                        / max(obs.loan_amount, 1),
                        2,
                    )
                    / 2
                )
                action = (
                    LoanAction.APPROVE
                    if rs > 0.55
                    else LoanAction.REJECT
                )
            else:
                # Easy: collateral heuristic
                action = (
                    LoanAction.APPROVE
                    if obs.mortgage_value >= 1.1 * obs.loan_amount
                    else LoanAction.REJECT
                )

            if task == "hard":
                _, reward, _, info = env.step(action, council_result=council)
            else:
                _, reward, _, info = env.step(action)
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

    print(f"\n{'='*60}")
    print(f"TOTAL REWARD: {total:.4f}")
    print(f"FINAL SCORE: {score:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_inference()
