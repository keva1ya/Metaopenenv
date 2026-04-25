"""
RL trace buffer for weighted multi-agent decisions.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class TraceRecord:
    """Single decision trace used for GRPO/offline training loops."""

    episode: int
    final_action: str
    ground_truth_label: Optional[str]
    reward_total: float
    reward_confidence: float
    reward_alignment: float
    uncertainty_cost_applied: float
    uncertainty_score: float
    penalty_mode: str
    penalty_config_version: str
    head_votes: Dict[str, str]
    head_confidences: Dict[str, float]
    head_reasoning: Dict[str, str]
    dynamic_weights: Dict[str, float]
    weighted_action_scores: Dict[str, float]
    diagnostics: Dict[str, float]


class TraceBuffer:
    """In-memory buffer with append-only JSONL persistence."""

    def __init__(
        self,
        trace_path: str = "logs/rl_trace.jsonl",
        max_buffer_size: int = 2048,
        flush_every: int = 32,
        enabled: bool = True,
    ):
        self.trace_path = trace_path
        self.max_buffer_size = max_buffer_size
        self.flush_every = flush_every
        self.enabled = enabled
        self._buffer: List[TraceRecord] = []
        self.total_written: int = 0

    def push(self, record: TraceRecord) -> None:
        if not self.enabled:
            return
        self._buffer.append(record)
        if len(self._buffer) >= self.flush_every or len(self._buffer) >= self.max_buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self.enabled or not self._buffer:
            return
        os.makedirs(os.path.dirname(self.trace_path), exist_ok=True)
        with open(self.trace_path, "a") as f:
            for rec in self._buffer:
                f.write(json.dumps(asdict(rec), ensure_ascii=True) + "\n")
                self.total_written += 1
        self._buffer.clear()

    def pending_count(self) -> int:
        return len(self._buffer)

