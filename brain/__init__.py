"""
Brain module: DQN agent, state builder, and reward shaper.

Heavy imports (torch-dependent) are not eagerly loaded.
Import specific classes directly from submodules:
  from brain.big_brain import BigBrainAgent
  from brain.state_builder import PortfolioState, build_state_vector
  from brain.reward_shaper import calculate_shaped_reward
"""

__all__ = [
    "BigBrainAgent",
    "BigBrainNetwork",
    "ExperienceReplayBuffer",
    "PortfolioState",
    "build_state_vector",
    "calculate_shaped_reward",
]
