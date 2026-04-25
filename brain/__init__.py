"""
Brain module: weighted decision engine, state builder, and reward shaper.

Heavy imports (torch-dependent) are not eagerly loaded.
Import specific classes directly from submodules.
"""

__all__ = [
    "DynamicWeightDecisionEngine",
    "DynamicWeightNet",
    "WeightedDecisionResult",
    "PortfolioState",
    "build_state_vector",
    "calculate_shaped_reward",
]
