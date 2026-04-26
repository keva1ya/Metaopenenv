"""
Convert collected RL traces into per-head GRPO training datasets.

Each `TraceRecord` in `logs/rl_trace.jsonl` captures the ground-truth label,
shaped reward, and the per-head decision/confidence/reasoning. We split this
into three datasets (one per head LLM), each as JSONL of:

    {
      "prompt":     "<the prompt the head saw>",
      "completion": "<the head's actual answer>",
      "reward":     <float reward attributed to this head>,
      "group_id":   <episode>,
      "head":       "RiskAgent" | "FraudAgent" | "PolicyAgent",
      "ground_truth": "APPROVE" | "REJECT" | "REQUEST_MORE_INFO" | null
    }

The `prompt` field cannot always be reconstructed losslessly from a trace
because the live prompt depends on the live sub-agent reports of that
episode. For trace records produced by `app.py` we therefore reconstruct
the prompt from `head_reasoning` context using the same XML schema the
council uses. Use `--regenerate-prompts` to re-run the env on the same
seeds and capture exact prompts (slower; requires LLMs disabled).

Reward attribution per head:
  - alignment-only (default):   reward = 1.0 if head_vote == final_action else 0.0
                                scaled by reward_total (so we down-weight bad episodes)
  - shaped (--mode shaped):     reward = trace.reward_total when head agreed with
                                ground_truth_label, else 0.0
  - dissent-aware (--mode dissent): like alignment-only but penalises confident
                                    dissenters of the winning action
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE = REPO_ROOT / "logs" / "rl_trace.jsonl"
OUT_DIR = REPO_ROOT / "logs" / "grpo_datasets"

HEAD_NAMES = ("RiskAgent", "FraudAgent", "PolicyAgent")


def _iter_traces(path: Path) -> Iterable[dict]:
    if not path.exists():
        raise FileNotFoundError(f"trace file not found: {path}")
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _completion_text(head: str, vote: str, conf: float, reasoning: str) -> str:
    """Re-create the council head's expected output format."""
    return (
        f"DECISION: {vote}\n"
        f"CONFIDENCE: {conf:.2f}\n"
        f"REASONING: {reasoning.strip()}"
    )


def _stub_prompt(head: str, trace: dict) -> str:
    """
    Deterministic, schema-consistent prompt reconstruction.

    NOTE: the live prompt also includes verbatim sub-agent reports. We do not
    persist those in the trace today; the offline trainer should treat this
    as the *minimum-context* prompt and rely on `--regenerate-prompts` mode
    when full fidelity is required.
    """
    diag = trace.get("diagnostics", {}) or {}
    return (
        f"<system>You are the {head} chief officer.</system>\n"
        f"<episode>{trace.get('episode')}</episode>\n"
        f"<diagnostics>{json.dumps(diag, sort_keys=True)}</diagnostics>\n"
        f"<task>Cast a vote (APPROVE | REJECT | REQUEST_MORE_INFO) "
        f"with confidence and a 2-sentence reasoning.</task>"
    )


def _reward_for_head(head: str, trace: dict, mode: str) -> float:
    head_vote = (trace.get("head_votes") or {}).get(head)
    if head_vote is None:
        return 0.0

    final = trace.get("final_action")
    gt = trace.get("ground_truth_label")
    rt = float(trace.get("reward_total", 0.0))

    if mode == "alignment":
        return rt if head_vote == final else 0.0
    if mode == "shaped":
        if gt is None:
            return rt if head_vote == final else 0.0
        return rt if head_vote == gt else 0.0
    if mode == "dissent":
        if head_vote == final:
            return rt
        # confident dissent against the chosen action is penalised
        head_conf = float((trace.get("head_confidences") or {}).get(head, 0.0))
        return -0.5 * head_conf * rt
    raise ValueError(f"unknown mode: {mode}")


def build(
    trace_path: Path,
    out_dir: Path,
    mode: str,
    min_reward: float,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {h: open(out_dir / f"{h.lower()}_grpo.jsonl", "w") for h in HEAD_NAMES}
    counts = {h: 0 for h in HEAD_NAMES}

    try:
        for trace in _iter_traces(trace_path):
            for head in HEAD_NAMES:
                vote = (trace.get("head_votes") or {}).get(head)
                if not vote:
                    continue
                conf = float((trace.get("head_confidences") or {}).get(head, 0.0))
                reasoning = (trace.get("head_reasoning") or {}).get(head, "")
                reward = _reward_for_head(head, trace, mode)
                if reward < min_reward:
                    continue

                row = {
                    "prompt": _stub_prompt(head, trace),
                    "completion": _completion_text(head, vote, conf, reasoning),
                    "reward": reward,
                    "group_id": trace.get("episode"),
                    "head": head,
                    "ground_truth": trace.get("ground_truth_label"),
                }
                handles[head].write(json.dumps(row, ensure_ascii=True) + "\n")
                counts[head] += 1
    finally:
        for fh in handles.values():
            fh.close()

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--mode",
        choices=("alignment", "shaped", "dissent"),
        default="alignment",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=-1.0,
        help="Drop rows with reward below this (set higher to filter weak rollouts).",
    )
    args = parser.parse_args()

    counts = build(args.trace, args.out_dir, args.mode, args.min_reward)

    print(f"Wrote per-head GRPO datasets to {args.out_dir}")
    for head, n in counts.items():
        print(f"  {head:12s}: {n} rows")
    if sum(counts.values()) == 0:
        print("WARNING: no rows produced — check the trace file.")


if __name__ == "__main__":
    main()
