"""
Sync local trace logs to a Hugging Face Dataset repository.

Designed to be run either:
  - as a periodic background thread inside the HF Space, OR
  - as a standalone cron job from a separate worker.

Uses the HfApi.upload_file path which performs idempotent, atomic uploads
(LFS-aware), so it is safe to call repeatedly with the same source paths.

Required env vars:
  HF_TOKEN                      Hugging Face write token
  HF_TRACE_DATASET_REPO         e.g. "your-username/metaopenenv-traces"

Optional:
  RL_TRACE_PATH                 default: logs/rl_trace.jsonl
  HF_TRACE_SYNC_INTERVAL_SEC    when --loop is passed; default 3600
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "huggingface_hub is required. Install with: pip install huggingface_hub"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILES = [
    "logs/rl_trace.jsonl",
    "logs/dissent_log.json",
]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sync_once(
    repo_id: str,
    token: str,
    files: list[str],
    archive: bool = False,
) -> None:
    """Upload each file to the dataset repo. If `archive`, also keep a timestamped copy."""
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)

    stamp = _utc_stamp()
    for rel in files:
        src = REPO_ROOT / rel
        if not src.exists():
            print(f"[skip] {src} does not exist yet")
            continue

        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=rel,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"sync {rel} @ {stamp}",
        )
        print(f"[ok]   {rel} → {repo_id}")

        if archive:
            archive_path = f"archive/{stamp}/{Path(rel).name}"
            api.upload_file(
                path_or_fileobj=str(src),
                path_in_repo=archive_path,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"archive {rel} @ {stamp}",
            )
            print(f"[ok]   {rel} → {archive_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument(
        "--interval-sec",
        type=int,
        default=int(os.getenv("HF_TRACE_SYNC_INTERVAL_SEC", "3600")),
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Also write timestamped archive copies",
    )
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    args = parser.parse_args()

    repo_id = os.getenv("HF_TRACE_DATASET_REPO")
    token = os.getenv("HF_TOKEN")
    if not repo_id or not token:
        print(
            "ERROR: HF_TRACE_DATASET_REPO and HF_TOKEN must be set.",
            file=sys.stderr,
        )
        sys.exit(2)

    while True:
        try:
            sync_once(repo_id=repo_id, token=token, files=args.files, archive=args.archive)
        except Exception as exc:  # noqa: BLE001
            print(f"[err]  sync failed: {exc}", file=sys.stderr)
        if not args.loop:
            return
        time.sleep(max(60, args.interval_sec))


if __name__ == "__main__":
    main()
