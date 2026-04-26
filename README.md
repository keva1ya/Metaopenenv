---
title: AI Loan Approval OpenEnv
emoji: 🏦
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# AI Loan Approval OpenEnv — Technical Reference

A **three-level, multi-agent loan underwriting environment** with an **OpenEnv-style** contract (see `openenv.yaml`): six rule-based specialist sub-agents, three **policy** sub-agents with optional LLM + rule fallbacks, three head LLMs (OpenAI-compatible APIs, typically [Hugging Face](https://huggingface.co) Inference Endpoints or the HF router), a **deterministic weighted aggregator** (default), shaped rewards, dissent logging, and **append-only** RL trace records for offline training and analytics.

| Layer | Role | Technology |
|-------|------|------------|
| **L1** | Domain checks | Rules for Risk/Fraud; **optional** policy LLM for Policy sub-agents (`agents/sub_agents/`) |
| **L2** | Synthesis & votes | `RiskAgent`, `FraudAgent`, `PolicyAgent` + optional LLM (`agents/council/`) |
| **L3** | Final action from votes × confidence × UI multipliers | `DynamicWeightDecisionEngine` (`brain/dynamic_weight_net.py`) |

**Problem in one line:** single scoring pipelines are opaque and monolithic; this system enforces **separation of concerns** (risk vs fraud vs policy), **auditable debate** (dissent), and **safe production defaults** (rule fallbacks, no online combiner training unless explicitly enabled).

---

## Table of contents

1. [OpenEnv & contract](#1-openenv--contract)
2. [Core data model](#2-core-data-model)
3. [End-to-end control flow](#3-end-to-end-control-flow)
4. [Level 1 — Specialist sub-agents](#4-level-1--specialist-sub-agents)
5. [Level 2 — Council head agents & LLM wiring](#5-level-2--council-head-agents--llm-wiring)
6. [Voting, ties, and dissent](#6-voting-ties-and-dissent)
7. [Level 3 — Weighted decision engine](#7-level-3--weighted-decision-engine)
8. [Reward shaping](#8-reward-shaping)
9. [RL traces & offline training](#9-rl-traces--offline-training)
10. [Security (summary)](#10-security-summary)
11. [Runtime configuration (environment variables)](#11-runtime-configuration-environment-variables)
12. [API (FastAPI)](#12-api-fastapi)
13. [User interface (Gradio)](#13-user-interface-gradio)
14. [Task levels & mortgage mode](#14-task-levels--mortgage-mode)
15. [Repository layout](#15-repository-layout)
16. [Installation & operations](#16-installation--operations)
17. [Hugging Face deployment](#17-hugging-face-deployment)
18. [Design rationale](#18-design-rationale)
19. [Further reading](#19-further-reading)

---

## 1. OpenEnv & contract

The machine-readable spec lives in [`openenv.yaml`](openenv.yaml).

| Field | Value |
|-------|--------|
| **Name** | `LoanApprovalOpenEnv` |
| **Tasks** | `easy`, `medium`, `hard` |
| **Modes** | `normal`, `mortgage_mode` (auto-approve when collateral threshold met; see [§14](#14-task-levels--mortgage-mode)) |
| **Action space** | `APPROVE`, `REJECT`, `REQUEST_MORE_INFO` |
| **Brain** | `weighted_moe`, state dim 28, action dim 3, `GRPO_compatible` training label |
| **Reward range (spec)** | `min: -1`, `max: 1` (shaped reward in code is usually clipped to `[0, 1]`; see [§8](#8-reward-shaping)) |

`hard` is the default in `app.py` and exercises the full L1 → L2 → L3 path.

---

## 2. Core data model

### 2.1 Observation (`env/observation.py`)

`LoanObservation` is a **Pydantic** model (frozen) with **18** fields. Key groupings:

- **Financial / structural:** `income`, `credit_score`, `employment_years`, `loan_amount`, `loan_type`, `mortgage_value`, `loan_term`, `existing_debt`, `age`, `monthly_expenses`, `num_dependents`, `previous_defaults`, `gst_registered`, `itr_filed_years`, `customer_score`.
- **Behavioral / integrity:** `spending_volatility`, `loan_purpose_clarity`, `application_consistency` (Fraud path emphasis).
- Ranges and defaults are defined via `Field(...)` constraints (e.g. `mortgage_value` ∈ [0, 700_000] per generator).

Synthetic applicants are produced by `SyntheticDataGenerator` in `env/data_generator.py`.

### 2.2 Actions (`env/action.py`)

`LoanAction` enum: `APPROVE`, `REJECT`, `REQUEST_MORE_INFO`. Used in the environment, API request bodies, and head-agent output parsing.

---

## 3. End-to-end control flow

1. **Reset** → new `LoanObservation`, portfolio state updated, episode counter advanced (`env/loan_env.py`).
2. **Mortgage fast path (optional)** → if `mortgage_value >= 1.3 × loan_amount`, environment can short-circuit to approve with `mode="mortgage_mode"` (no council required for the heavy path in that branch — see `LoanApprovalEnv.step` logic).
3. **Otherwise (hard / normal)** → run nine sub-agents, collect **structured reports** per domain.
4. **Council** → `RiskAgent`, `FraudAgent`, `PolicyAgent` each receive domain reports + **XML-delimited** context (prompt isolation) and return **decision, confidence, reasoning**; on LLM failure, **deterministic rules** in the same agent classes apply.
5. **VotingEngine** → majority vote, **3-way tie broken by `PolicyAgent`**, dissent entries recorded (`agents/voting.py`).
6. **DynamicWeightDecisionEngine** → combines head votes with **per-head confidence** and **UI manual multipliers**; produces final `LoanAction` and per-action score vector (`brain/dynamic_weight_net.py`). Neural branch exists but is **off by default** (see [§7](#7-level-3--weighted-decision-engine)).
7. **Reward** → `calculate_shaped_reward(obs, action, task_level)` (`brain/reward_shaper.py`).
8. **Trace** → if enabled, a `TraceRecord` is pushed to `TraceBuffer` and flushed to `RL_TRACE_PATH` as JSONL (`brain/trace_buffer.py`, wired in `app.py` / env).

---

## 4. Level 1 — Specialist sub-agents

All nine agents are **deterministic** (no network calls). They return recommendations, confidence, and flags consumed by the council.

| Domain | Class | File | Notes |
|--------|--------|------|--------|
| Risk | `CreditAnalystAgent` | `agents/sub_agents/risk/credit_analyst.py` | Rule-based only |
| Risk | `IncomeVerifierAgent` | `agents/sub_agents/risk/income_verifier.py` | Rule-based only |
| Risk | `DebtAnalyzerAgent` | `agents/sub_agents/risk/debt_analyzer.py` | Rule-based only |
| Fraud | `PatternCheckerAgent` | `agents/sub_agents/fraud/pattern_checker.py` | Rule-based only |
| Fraud | `ConsistencyCheckerAgent` | `agents/sub_agents/fraud/consistency_checker.py` | Rule-based only |
| Fraud | `IdentityVerifierAgent` | `agents/sub_agents/fraud/identity_verifier.py` | Rule-based only |
| Policy | `ComplianceOfficerAgent` | `agents/sub_agents/policy/compliance_officer.py` | **Optional LLM** when a client is passed |
| Policy | `TermValidatorAgent` | `agents/sub_agents/policy/term_validator.py` | **Optional LLM** when a client is passed |
| Policy | `LTVCheckerAgent` | `agents/sub_agents/policy/ltv_checker.py` | **Optional LLM** when a client is passed |

**Policy desk:** `LoanApprovalEnv` passes the same **policy** OpenAI-compatible client and `model_name` (from `POLICY_*` / `HF_TOKEN` resolution) into these three classes. If `client is None` (e.g. no API key), they use their **rule-only `_fallback`/`_rule_based`** paths — no network calls.

---

## 5. Level 2 — Council head agents & LLM wiring

| Head | Role | Default model line (configurable) |
|------|------|--------------------------------------|
| `RiskAgent` | Repayment & capacity | e.g. Qwen-family fine-tune on HF |
| `PolicyAgent` | Rules & compliance | e.g. Mistral fine-tune |
| `FraudAgent` | Behavioural / integrity | e.g. Llama 3 fine-tune |

Implementation: `agents/council/risk_agent.py`, `fraud_agent.py`, `policy_agent.py`. Each uses the **OpenAI** Python client with a **configurable** `base_url` + `api_key` + `model` string.

### 5.1 Credential & endpoint resolution

In `env/loan_env.py`, per head:

1. If `<AGENT>_API_KEY` and `<AGENT>_MODEL` are set → use them with `<AGENT>_API_BASE` or the default Hugging Face OpenAI-compatible router base.
2. Else if `HF_TOKEN` and `<AGENT>_MODEL` are set → use shared token (same default base if base unset).
3. Else if legacy `NEMOTRON_API_KEY` / `GLM_API_KEY` / `MINIMAX_API_KEY` (or `OPENROUTER_API_KEY`) are set → map to **OpenRouter** `base_url` and default model IDs for that legacy path.
4. Else **demo mode** for that head: rule-based fallback only.

`app.py` sets `DEMO_MODE` when **all three** head paths are unconfigured (see `_agent_configured` helpers).

### 5.2 Prompting & fallbacks

- User-facing **data** sections use **XML-style tags** (e.g. `<sub_agent_reports>`, `<applicant_data>`, `<dissent_history>`, `<identity_guidance>`) to reduce prompt-injection surface.
- On any LLM error, the head agent falls back to **rule evaluation** in Python so the environment never hard-fails on a timeout.

**Integration details, diagrams, and validation checklist:** [`docs/model_integration_guide.md`](docs/model_integration_guide.md).

---

## 6. Voting, ties, and dissent

**`VotingEngine`** (`agents/voting.py`):

- Aggregates the three `Vote` objects.
- **Majority** wins; if all three differ (**3-way tie**), the **`PolicyAgent` vote** breaks the tie (see source comments: “3-way tie: PolicyAgent breaks tie”).
- **Dissent:** any head whose decision ≠ majority gets logged with a **sanitized** reason (length cap + pattern stripping) before persistence.

Dissent and related metadata are written to `logs/dissent_log.json` (contents gitignored; path may vary with deployment).

---

## 7. Level 3 — Weighted decision engine

**Class:** `DynamicWeightDecisionEngine` in `brain/dynamic_weight_net.py`.

It consumes:

- Current `LoanObservation` (for state features via `state_builder.py`),
- A `VoteResult` from the council (votes + per-head confidence),
- `PortfolioState` (approved / rejected / reward history, etc.),
- **Manual multipliers** from the UI, keyed by `RiskAgent`, `FraudAgent`, `PolicyAgent`.

### 7.1 Default path: manual × confidence (aggregator-only)

With `DYNAMIC_WEIGHT_USE_NEURAL=false` (default in `app.py`):

- Raw weight vector = **confidence[agent] × manual_multiplier[agent]** (clamped non-negative), then **normalized** to sum to 1.
- Each head’s vote adds its normalized weight to the score of that action (APPROVE / REJECT / REQUEST_MORE_INFO).
- Final action = **argmax** over the 3-way action score vector; an **uncertainty** scalar is derived from the spread of that distribution.

**No online gradient** is applied to any PyTorch module in this path; the `DynamicWeightNet` module is instantiated but not used for forward passes when `use_neural_weights=False`.

### 7.2 Optional path: neural weights

If `DYNAMIC_WEIGHT_USE_NEURAL=true`, a small MLP may produce softmax weights over the three heads (requires PyTorch; still combined with the same vote-aggregation idea — see `decide` implementation). This is **not** the recommended production default for the “aggregator-only” deployment.

---

## 8. Reward shaping

**Function:** `calculate_shaped_reward` in `brain/reward_shaper.py`.

- **Hard overrides (immediate 0.0 for dangerous APPROVE):**
  - Strong fraud heuristics (see `_is_strong_fraud`) and APPROVE.
  - High financial risk (composite score &lt; 0.3) and APPROVE.
- **Mortgage shortcut:** if `mortgage_value >= 1.3 × loan_amount`, reward is **1.0** if action is APPROVE, else **0.0** (aligns with `mortgage_mode` semantics).
- **Else:**  
  `reward = 0.4 × r_financial + 0.4 × r_fraud + 0.2 × r_policy`, then **clipped** to `[0, 1]`.

Component definitions are in private helpers `_financial_reward`, `_fraud_reward`, `_policy_reward`. This reward is used for **monitoring, UI, and trace logging**; it does **not** by itself update model weights in production when `GRPO_TRAIN_ENABLED=false`.

There is a thin `env/reward.py` wrapper used where the grader or env expects a small facade — see call sites in `env/loan_env.py` and `env/grader.py`.

---

## 9. RL traces & offline training

### 9.1 `TraceRecord` (`brain/trace_buffer.py`)

Each record includes: `episode`, `final_action`, `ground_truth_label` (if any), `reward_total` and decomposed reward fields, per-head votes/confidences/reasoning, normalised dynamic weights, weighted action scores, and a `diagnostics` dict. Records are **append-only JSONL** at `RL_TRACE_PATH` (default `logs/rl_trace.jsonl`).

**Environment flags:** `RL_TRACE_ENABLED`, `RL_TRACE_PATH`, `RL_TRACE_FLUSH_EVERY`, `RL_TRACE_MAX_BUFFER`.

### 9.2 GRPO scaffold (no online training by default)

`GRPOTrainerScaffold` in `brain/grpo_trainer.py` is a **placeholder**: when `GRPO_TRAIN_ENABLED=false` it never performs optimisation steps. It exists to keep the cadence and interfaces ready for future labelled data.

### 9.3 Offline pipeline (separate from the Space)

| Script | Purpose |
|--------|--------|
| `scripts/prepare_grpo_dataset.py` | Build per-head JSONL from traces (reward modes: alignment / shaped / dissent). |
| `scripts/train_grpo_offline.py` | Optional TRL GRPO-style training (heavy deps: see `requirements-train.txt`). |
| `scripts/sync_logs_to_hf_dataset.py` | Push `rl_trace.jsonl` / `dissent_log.json` to a HF **Dataset** repo. |
| `scripts/inference.py` | Benchmarks easy/medium/hard trajectories; uses deterministic engine for `hard` in script as written. |
| `scripts/test_offline.py` | Architecture smoke test without requiring live LLMs. |
| `scripts/smoke_test_deployment.py` | Health + local aggregation + trace + fallback checks. |

---

## 10. Security (summary)

- **FastAPI:** optional `X-API-Key` when `APP_API_KEY` is set; rate limits on routes; CORS (tighten in production); `StepRequest` validation.
- **Gradio:** optional HTTP basic auth when `APP_UI_USERNAME` / `APP_UI_PASSWORD` are set.
- **LLM layer:** XML delimiters in prompts; dissent reasoning sanitisation.  
Full detail: [`docs/SECURITY_ARCHITECTURE.md`](docs/SECURITY_ARCHITECTURE.md).

---

## 11. Runtime configuration (environment variables)

**Aggregator & training**

| Variable | Default | Meaning |
|----------|---------|--------|
| `DYNAMIC_WEIGHT_USE_NEURAL` | `false` | Use only manual×confidence weighting. |
| `GRPO_TRAIN_ENABLED` | `false` | Disable online GRPO placeholder training. |
| `GRPO_TRAIN_EVERY_STEPS` | `16` | Cadence (only if enabled). |
| `GRPO_MIN_TRACE_BEFORE_TRAIN` | `256` | Min traces (only if enabled). |
| `REWARD_FLOOR` | `0.0` | Passed into GRPO config. |
| `UNCERTAINTY_PENALTY_ENABLED` | `false` | Placeholder for uncertainty penalty. |
| `UNCERTAINTY_PENALTY_WEIGHT` | `0.0` | Weight for uncertainty term. |
| `REWARD_MODE` | `non_negative_v1` | Label in GRPO config. |

**Tracing**

| Variable | Default | Meaning |
|----------|---------|--------|
| `RL_TRACE_ENABLED` | `true` | Master switch for `TraceBuffer`. |
| `RL_TRACE_PATH` | `logs/rl_trace.jsonl` | JSONL path. |
| `RL_TRACE_FLUSH_EVERY` | `32` | Flush after N records. |
| `RL_TRACE_MAX_BUFFER` | `2048` | In-memory cap before forced flush. |

**Head models (per agent, prefix `RISK` / `FRAUD` / `POLICY`)**

- `<P>_MODEL` — model id string (e.g. `org/model-name`).
- `<P>_API_KEY` — optional; else `HF_TOKEN` for that agent.
- `<P>_API_BASE` — optional OpenAI-compatible base (HF Endpoint URL ending in `/v1/` or router).

Also: `HF_TOKEN`, and legacy `NEMOTRON_API_KEY` / `GLM_API_KEY` / `MINIMAX_API_KEY`, `OPENROUTER_API_KEY` as in `loan_env.py`.

**Service ports & auth**

| Variable | Role |
|----------|------|
| `APP_API_PORT` | FastAPI (default 8000). |
| `APP_UI_PORT` | Gradio (default 7860). |
| `APP_API_KEY` | If set, required on `/reset`, `/step`, `/state` as `X-API-Key`. |
| `APP_UI_USERNAME` / `APP_UI_PASSWORD` | If both set, Gradio basic auth. |

**Log sync to Hub (optional)**  
`HF_TRACE_DATASET_REPO`, `HF_TRACE_SYNC_INTERVAL_SEC` — used by `scripts/sync_logs_to_hf_dataset.py`.

A copy-paste template is in [`.env.example`](.env.example). **Do not commit real secrets;** use Space secrets and local `.env` (`.env` is gitignored).

---

## 12. API (FastAPI)

Base URL: `http://<host>:<APP_API_PORT>` (default `127.0.0.1:8000`). CORS is currently permissive — restrict origins in production.

| Endpoint | Method | Auth | Rate limit (default) | Description |
|----------|--------|------|----------------------|-------------|
| `/reset` | POST | If `APP_API_KEY` set | 10/min | New episode: fresh observation JSON. |
| `/step` | POST | If `APP_API_KEY` set | 30/min | One env step. Body: `{"action": "APPROVE" \| "REJECT" \| "REQUEST_MORE_INFO"}`. |
| `/state` | GET | If `APP_API_KEY` set | 60/min | Current observation or error if not reset. |
| `/health` | GET | No | 60/min | `status`, `episode`, `demo_mode`. |

**Note:** The Gradio UI drives the same environment in-process; REST clients use the same global `LoanApprovalEnv` instance. When `APP_API_KEY` is unset, the dependency allows unauthenticated access (suitable for local dev only).

---

## 13. User interface (Gradio)

`app.py` runs **Gradio** (`Blocks`) and, in a **background thread**, **Uvicorn** for FastAPI. Features include applicant profile, sub-agent accordions, council votes, weighted decision, portfolio metrics, dissent log, multipliers, optional skeleton state during long operations, and a demo banner when `DEMO_MODE` is true.

---

## 14. Task levels & mortgage mode

| Task | Grading / behaviour (summary) |
|------|---------------------------------|
| `easy` | `EasyGrader`: collateral check — **APPROVE** if `mortgage_value >= 1.3 × loan_amount` else **REJECT** (`env/grader.py`). |
| `medium` | `MediumGrader`: composite risk score vs threshold (~0.55). |
| `hard` | `HardGrader` + full pipeline with weighted engine in production UI. |

**Mortgage mode:** when `mortgage_value >= 1.3 × loan_amount`, the environment can take the fast **auto-approval** path (`mode="mortgage_mode"` in info) consistent with reward shaper and UI copy in `app.py` (1.3×, not 1.1×).

---

## 15. Repository layout

```text
Metaopenenv/
├── app.py                      # Gradio UI + Uvicorn + FastAPI + trace wiring
├── openenv.yaml                # OpenEnv contract
├── requirements.txt            # App + API + Gradio
├── requirements-train.txt      # Offline TRL/GRPO training (do not use on Space)
├── .env.example                # Non-secret template
├── agents/
│   ├── council/                # risk_agent, fraud_agent, policy_agent
│   ├── sub_agents/             # risk / fraud / policy specialists
│   └── voting.py
├── brain/
│   ├── dynamic_weight_net.py   # Final weighted decision + optional neural branch
│   ├── reward_shaper.py
│   ├── state_builder.py         # 28-dim state for RL / combiner
│   ├── trace_buffer.py
│   └── grpo_trainer.py
├── env/
│   ├── loan_env.py
│   ├── observation.py, action.py, data_generator.py
│   ├── grader.py, reward.py
├── scripts/
│   ├── inference.py, test_offline.py, scratch.py
│   ├── prepare_grpo_dataset.py, train_grpo_offline.py
│   ├── sync_logs_to_hf_dataset.py, smoke_test_deployment.py
├── logs/                       # Runtime logs (gitignore patterns apply)
└── docs/
    ├── HUGGINGFACE_DEPLOYMENT.md
    ├── model_integration_guide.md
    ├── SECURITY_ARCHITECTURE.md
    └── IMPLEMENTATION_GUIDE.md
```

---

## 16. Installation & operations

```bash
cd Metaopenenv
python -m venv .venv
source .venv/bin/activate   # or Windows equivalent
pip install -r requirements.txt
cp .env.example .env         # then edit secrets
python app.py                # Gradio + API
```

**Verify stack without API keys:** `python scripts/test_offline.py`  
**Benchmark:** `python scripts/inference.py`

**Offline training (separate machine):** `pip install -r requirements-train.txt` then follow [`docs/HUGGINGFACE_DEPLOYMENT.md`](docs/HUGGINGFACE_DEPLOYMENT.md) and the scripts’ `--help` output.

---

## 17. Hugging Face deployment

- **Runbook (aggregator-only Space, three Inference Endpoints, trace sync):** [`docs/HUGGINGFACE_DEPLOYMENT.md`](docs/HUGGINGFACE_DEPLOYMENT.md)
- **Pre-launch script:** `scripts/smoke_test_deployment.py`
- The README **YAML frontmatter** at the top of this file configures **HF Spaces** (`sdk: gradio`, `app_file: app.py`).

---

## 18. Design rationale

- **Why three head models?** Domain-specialised fine-tunes reduce single-model blind spots; each head only sees the reports and features relevant to risk, fraud, or policy.
- **Why rule-based sub-agents (Risk/Fraud) and optional LLM (Policy subs)?** Risk and fraud desks stay **deterministic** for speed and auditability. Policy sub-agents can use the **same** policy model as `PolicyAgent` for nuanced wording, with **rule fallbacks** if the client is missing or the call fails.
- **Why deterministic aggregator by default?** Production decisions stay **explainable** (multipliers × confidence); optional neural combiner is for experimentation only.
- **Why shaped reward + hard overrides?** To penalise **bad APPROVEs** on fraud and financial grounds before component scores are blended.
- **Why traces?** For **offline** GRPO / preference training on head models without coupling training to the live request path.

---

## 19. Further reading

| Document | Content |
|----------|---------|
| [`docs/model_integration_guide.md`](docs/model_integration_guide.md) | Endpoints, env vars, prompt shapes, Mermaid architecture |
| [`docs/HUGGINGFACE_DEPLOYMENT.md`](docs/HUGGINGFACE_DEPLOYMENT.md) | Phased deployment, env matrix, smoke tests |
| [`docs/SECURITY_ARCHITECTURE.md`](docs/SECURITY_ARCHITECTURE.md) | API, UI, LLM, secrets |
| [`docs/IMPLEMENTATION_GUIDE.md`](docs/IMPLEMENTATION_GUIDE.md) | Additional implementation notes |

---

## Acknowledgements

[Gradio](https://gradio.app), [FastAPI](https://fastapi.tiangolo.com), [Pydantic](https://docs.pydantic.dev), [PyTorch](https://pytorch.org) (for optional `DynamicWeightNet` / training scripts), [Hugging Face](https://huggingface.co) Hub and Inference, and [OpenAI](https://platform.openai.com)-compatible client libraries for head endpoints.

---

*This README is the canonical technical overview for the repository; narrative “story” copy was consolidated into this structure for evaluators, operators, and contributors who need exact behaviour and file-level pointers.*
