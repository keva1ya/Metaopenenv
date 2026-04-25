"""
GRPO trainer scaffold.

This module intentionally keeps training side effects minimal until
external labeled data is integrated.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from brain.trace_buffer import TraceRecord


@dataclass
class GRPOConfig:
    enabled: bool = False
    train_every_steps: int = 16
    min_trace_before_train: int = 256
    group_size: int = 8
    reward_floor: float = 0.0
    uncertainty_penalty_enabled: bool = False
    uncertainty_penalty_weight: float = 0.0
    reward_mode: str = "non_negative_v1"


class GRPOTrainerScaffold:
    """
    Non-invasive GRPO training harness.

    Current behavior:
      - Tracks step cadence.
      - Collects trace batches.
      - Skips optimization when labels are unavailable.
    """

    def __init__(self, config: GRPOConfig):
        self.config = config
        self.update_steps = 0
        self.last_status = "idle"

    def load_labels_for_batch(
        self, trace_ids: Sequence[int]
    ) -> Dict[int, Optional[str]]:
        """
        Adapter placeholder for external dataset labels.

        Replace this with dataset-backed lookup when labels are ready.
        """
        return {tid: None for tid in trace_ids}

    def maybe_train(self, traces: List[TraceRecord]) -> str:
        if not self.config.enabled:
            self.last_status = "disabled"
            return self.last_status
        if len(traces) < self.config.min_trace_before_train:
            self.last_status = "warmup"
            return self.last_status
        if self.update_steps % self.config.train_every_steps != 0:
            self.last_status = "waiting_interval"
            self.update_steps += 1
            return self.last_status

        trace_ids = list(range(len(traces)))
        labels = self.load_labels_for_batch(trace_ids)
        if all(v is None for v in labels.values()):
            self.last_status = "skipped_no_labels"
            self.update_steps += 1
            return self.last_status

        # Future GRPO update logic will be inserted here.
        self.last_status = "trained_placeholder"
        self.update_steps += 1
        return self.last_status

