# Hugging Face Deployment — Aggregator-Only Runbook

This runbook ships **Metaopenenv** to a Hugging Face Space as a deterministic,
non-learning aggregator. The three head LLMs (Qwen / Mistral / Llama 3) are
hosted as **separate Hugging Face Inference Endpoints**. The Space hosts only
the orchestration layer (`app.py`, environment, sub-agents, voting, weighted
aggregator, trace logging).

> The combiner does **no online learning** in production. RL traces are
> collected purely for offline analytics and future head-model fine-tuning.

---

## Phase 1 — Repository & Hosting Architecture

| Component        | Where it lives                                        | Notes |
|------------------|-------------------------------------------------------|-------|
| Orchestration    | HF Space (this repo, `app.py`, `sdk: gradio`)         | Frontmatter already present in `README.md` |
| Risk LLM         | HF Inference Endpoint (Qwen 2.5)                      | Owned by `RISK_API_BASE` |
| Policy LLM       | HF Inference Endpoint (Mistral 7B)                    | Owned by `POLICY_API_BASE` |
| Fraud LLM        | HF Inference Endpoint (Llama 3)                       | Owned by `FRAUD_API_BASE` |
| RL traces        | `logs/rl_trace.jsonl` → synced to HF **Dataset** repo | Persists across Space restarts |
| Dissent log      | `logs/dissent_log.json` → synced to HF Dataset repo   | Persists across Space restarts |
| Model adapters   | HF **Model** repos (one per head)                     | Versioned independently of the Space |

### 1.1 Create the Space

1. New Space → **SDK: Gradio**, hardware: CPU Basic is enough (no local LLM weights).
2. Connect this Git repo, OR push directly: `git remote add space https://huggingface.co/spaces/<user>/metaopenenv && git push space main`.
3. The `README.md` frontmatter already declares `sdk: gradio`, `app_file: app.py`.

### 1.2 Create the three Inference Endpoints

For each head model:

1. Push the merged adapter to a HF Model repo, e.g. `your-username/qwen-credit-risk`.
2. From that repo → **Deploy → Inference Endpoint** (or use a Dedicated Endpoint).
3. Choose a GPU appropriate for 7B/8B params (e.g. `nvidia-l4`).
4. Note the endpoint URL — use **the OpenAI-compatible `/v1/` URL**, e.g.
   `https://abcd1234.us-east-1.aws.endpoints.huggingface.cloud/v1/`.

### 1.3 Create the Dataset repo for logs

1. New Dataset repo: `your-username/metaopenenv-traces` (private).
2. The Space will push `logs/rl_trace.jsonl` and `logs/dissent_log.json` to it
   on a cron / scheduled job (see `scripts/sync_logs_to_hf_dataset.py`).

---

## Phase 2 — Runtime Configuration

Set these in **Space → Settings → Variables and secrets**.
Anything marked *secret* must be added via the **Secrets** tab, not Variables.

```env
# ── 1. Aggregator-only switches ────────────────────────────────
DYNAMIC_WEIGHT_USE_NEURAL=false   # combiner is deterministic
GRPO_TRAIN_ENABLED=false          # no online training in prod

# ── 2. Trace collection (for offline RL) ───────────────────────
RL_TRACE_ENABLED=true
RL_TRACE_PATH=logs/rl_trace.jsonl
RL_TRACE_FLUSH_EVERY=32

# ── 3. Head LLM endpoints ──────────────────────────────────────
HF_TOKEN=hf_********                                # SECRET
RISK_MODEL=your-username/qwen-credit-risk
RISK_API_BASE=https://<qwen>.endpoints.huggingface.cloud/v1/
POLICY_MODEL=your-username/mistral-policy-officer
POLICY_API_BASE=https://<mistral>.endpoints.huggingface.cloud/v1/
FRAUD_MODEL=your-username/llama3-fraud-detector
FRAUD_API_BASE=https://<llama>.endpoints.huggingface.cloud/v1/

# ── 4. UI + API auth ───────────────────────────────────────────
APP_UI_USERNAME=admin
APP_UI_PASSWORD=********                            # SECRET
APP_API_KEY=********                                # SECRET (FastAPI X-API-Key)

# ── 5. Optional: dataset sync target ───────────────────────────
HF_TRACE_DATASET_REPO=your-username/metaopenenv-traces
HF_TRACE_SYNC_INTERVAL_SEC=3600
```

> **Per-agent override:** if a particular endpoint requires a different
> token (e.g. third-party hosted), set `RISK_API_KEY` / `POLICY_API_KEY` /
> `FRAUD_API_KEY`. They take precedence over `HF_TOKEN` for that agent.

A complete copy-pasteable template lives in `.env.example` at the repo root.

---

## Phase 3 — Execution Flow (per applicant click)

```
User clicks "Next Applicant"
        │
        ▼
LoanApprovalEnv.reset()                         env/loan_env.py
        │  generates synthetic LoanObservation
        ▼
9 deterministic sub-agents run                  agents/specialist/*.py
        │  9 SubAgentReports (rec / confidence / flags)
        ▼
3 Head LLMs called over HTTPS                   agents/council/*.py
        │  Qwen → RISK_API_BASE
        │  Mistral → POLICY_API_BASE
        │  Llama 3 → FRAUD_API_BASE
        ▼
VotingEngine.vote(...)                          agents/voting.py
        │  vote_breakdown + agent_confidence + dissent
        ▼
DynamicWeightDecisionEngine.decide(...)         brain/dynamic_weight_net.py
        │  use_neural_weights=False  →  pure aggregator
        │  raw = confidence × manual_multiplier  (per agent)
        │  normalize → argmax over APPROVE/REJECT/REQUEST_MORE_INFO
        ▼
RewardShaper.calculate_shaped_reward(...)       brain/reward_shaper.py
        │  hard overrides (strong fraud / high risk) → 0.0
        │  else 0.4·r_fin + 0.4·r_fraud + 0.2·r_policy
        ▼
TraceBuffer.push(TraceRecord)                   brain/trace_buffer.py
        │  appended to logs/rl_trace.jsonl every 32 records
        ▼
DissentLog updated                              agents/voting.py
```

The combiner has **zero learnable parameters in the live runtime**. The
`DynamicWeightNet` torch module is instantiated but never called when
`DYNAMIC_WEIGHT_USE_NEURAL=false`.

---

## Phase 4 — Offline RL Lifecycle

The training loop is fully decoupled from the live Space.

```
┌──────────────────────────┐
│ 1. COLLECT (production)  │  Space appends rl_trace.jsonl
└────────────┬─────────────┘
             │ scripts/sync_logs_to_hf_dataset.py  (scheduled)
             ▼
┌──────────────────────────┐
│ 2. PREPARE (offline GPU) │  scripts/prepare_grpo_dataset.py
│   trace.jsonl  →  per-head prompt/completion/reward groups
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 3. TRAIN (offline GPU)   │  scripts/train_grpo_offline.py
│   GRPO via TRL on each head model independently
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 4. EVALUATE              │  scripts/inference.py + held-out trace eval
│   reward uplift, dissent stability, policy violation rate
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 5. DEPLOY                │  Push merged adapter to HF Model repo
│   Update Inference Endpoint → Space picks up automatically
└──────────────────────────┘
```

Step 5 requires **no Space redeploy** — the Space only knows the endpoint URL.

---

## Phase 5 — Pre-Launch Smoke Test Checklist

Run `python scripts/smoke_test_deployment.py --base-url <space-url>` against
the deployed Space. The script automates the four checks below.

| # | Check                  | Pass criterion |
|---|------------------------|----------------|
| 1 | API health             | `GET /health` → `{ "status": "ok", "demo_mode": false }` |
| 2 | Aggregation integrity  | Setting Risk multiplier to 100 and Fraud/Policy to 0 forces final action == Risk vote on ≥ 9/10 episodes |
| 3 | Trace logging          | A new line appended to `logs/rl_trace.jsonl` per `/step` call |
| 4 | Fallback safety        | Bad `HF_TOKEN` → no 5xx, sub-agents still run, council uses deterministic fallback |

Manual UI checks:

- [ ] Multipliers slider visibly changes the weighted bar.
- [ ] Dissent log card increments when heads disagree.
- [ ] Skeleton screen renders during processing.
- [ ] Basic auth gate on the Gradio UI works with `APP_UI_USERNAME` / `APP_UI_PASSWORD`.

---

## Acceptance Criteria

1. Combiner is deterministic and non-learning in production (`DYNAMIC_WEIGHT_USE_NEURAL=false`, `GRPO_TRAIN_ENABLED=false`).
2. Reward is captured every episode via `TraceBuffer` and persisted to the HF Dataset.
3. Head-model update lifecycle is fully offline and gated through `scripts/`.
4. The Space runs without local training jobs or local model weights.
5. OpenEnv contract (`openenv.yaml`) — tasks, action space, reward range — is unchanged.

---

## Operational notes

- **Cold-start of HF Inference Endpoints**: scale-to-zero endpoints add ~30s
  latency on the first request. Either disable scale-to-zero or set a
  keep-warm cron hitting `/health` of each endpoint every 10 minutes.
- **Cost control**: each `/step` triggers up to 3 LLM calls. Rate limiting on
  `/step` (30 req/min, configured in `app.py`) caps spend.
- **Secrets rotation**: rotating `HF_TOKEN` is a no-op restart of the Space —
  no code change required.
- **Rollback**: previous head models live in HF Model repos as commits;
  re-point the Inference Endpoint at the previous commit SHA to roll back
  without touching the Space.
