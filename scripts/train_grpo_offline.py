"""
Offline GRPO training entrypoint for a single head model.

Run separately for each head (Risk / Policy / Fraud). Designed for an
external GPU rig — NOT for the live Hugging Face Space.

Example:
  python scripts/train_grpo_offline.py \\
      --head RiskAgent \\
      --base-model your-username/qwen-credit-risk \\
      --dataset logs/grpo_datasets/riskagent_grpo.jsonl \\
      --output-dir checkpoints/risk_grpo_v1 \\
      --push-to-hub your-username/qwen-credit-risk-rlv1

Pipeline:
  1. Load base model in 4-bit, attach LoRA.
  2. Wrap dataset rows so TRL's GRPOTrainer sees prompt + scalar reward.
  3. Train for `--steps` GRPO steps with grouped advantage estimation.
  4. Merge LoRA → push merged model to HF Hub (optional).

This file is deliberately a thin, well-commented scaffold. The minute
hyper-parameters (batch size, lr, beta, max_new_tokens) are placeholders —
the user is expected to fill these in based on the head model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _lazy_imports():
    """Import heavy deps lazily so `--help` works in CPU-only envs."""
    import torch  # noqa: F401
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "trl >= 0.11 is required for GRPO. Install: pip install -U trl"
        ) from exc

    return {
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
    }


def load_dataset_jsonl(path: Path):
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_reward_fn(rows: list[dict]):
    """
    GRPOTrainer expects a callable: (prompts, completions, **kwargs) -> list[float].
    We pre-index the trace rewards by (prompt, completion) so the trainer can
    look them up — for fully-online reward you'd implement a live scoring fn.
    """
    table: dict[tuple[str, str], float] = {}
    for r in rows:
        table[(r["prompt"], r["completion"])] = float(r["reward"])

    def reward_fn(prompts, completions, **_kwargs):
        out = []
        for p, c in zip(prompts, completions):
            out.append(table.get((p, c), 0.0))
        return out

    return reward_fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--head", required=True, choices=("RiskAgent", "PolicyAgent", "FraudAgent"))
    ap.add_argument("--base-model", required=True, help="HF model id or local path")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--push-to-hub", default=None, help="HF repo id to push merged adapter to")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04, help="GRPO KL beta")
    ap.add_argument("--num-generations", type=int, default=4, help="GRPO group size")
    ap.add_argument("--max-prompt-length", type=int, default=1024)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--per-device-train-batch-size", type=int, default=2)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    deps = _lazy_imports()
    rows = load_dataset_jsonl(args.dataset)
    if not rows:
        raise SystemExit(f"empty dataset: {args.dataset}")
    print(f"loaded {len(rows)} GRPO rows for {args.head}")

    tokenizer = deps["AutoTokenizer"].from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"device_map": "auto"}
    if not args.no_4bit:
        model_kwargs["quantization_config"] = deps["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="float16",
        )
    model = deps["AutoModelForCausalLM"].from_pretrained(args.base_model, **model_kwargs)

    peft_config = deps["LoraConfig"](
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    ds = deps["Dataset"].from_list([{"prompt": r["prompt"]} for r in rows])

    grpo_cfg = deps["GRPOConfig"](
        output_dir=str(args.output_dir),
        learning_rate=args.lr,
        beta=args.beta,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.steps,
        logging_steps=10,
        save_steps=max(args.steps // 4, 1),
        bf16=True,
        report_to="none",
        push_to_hub=bool(args.push_to_hub),
        hub_model_id=args.push_to_hub,
    )

    reward_fn = build_reward_fn(rows)

    trainer = deps["GRPOTrainer"](
        model=model,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))

    if args.push_to_hub:
        trainer.push_to_hub(commit_message=f"GRPO offline run for {args.head}")
        print(f"pushed to {args.push_to_hub}")


if __name__ == "__main__":
    main()
