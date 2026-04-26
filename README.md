---
title: AI Loan Approval OpenEnv
emoji: 🏦
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# AI Loan Approval OpenEnv

A production-style multi-agent loan approval simulator with:
- deterministic specialist sub-agents,
- LLM-powered council synthesis with rule fallback, and
- a weighted final decision engine with trace-based training hooks.

The app starts both:
- Gradio UI on `http://127.0.0.1:7860`
- FastAPI backend on `http://127.0.0.1:8000`

## Architecture

### Level 1: Specialist sub-agents (deterministic)
Nine sub-agents evaluate one applicant:
- Risk: `CreditAnalyst`, `IncomeVerifier`, `DebtAnalyzer`
- Fraud: `PatternChecker`, `ConsistencyChecker`, `IdentityVerifier`
- Policy: `ComplianceOfficer`, `TermValidator`, `LTVChecker`

Each emits recommendation, confidence, and key flags.

### Level 2: Council head agents
Three head agents synthesize sub-agent reports:
- `RiskAgent`
- `FraudAgent`
- `PolicyAgent`

They use OpenRouter-backed models when keys are configured. If any LLM call fails (or keys are absent), they fall back to deterministic rule logic.

### Level 3: Final weighted engine
`DynamicWeightDecisionEngine` combines council outputs and portfolio state to produce the final action:
- `APPROVE`
- `REJECT`
- `REQUEST_MORE_INFO`

In hard mode, this level is the final decision point.

## Repository Layout

```text
Metaopenenv/
├── app.py
├── inference.py
├── test_offline.py
├── openenv.yaml
├── requirements.txt
├── agents/
├── brain/
├── env/
├── docs/
└── logs/
```

## Setup

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Create `.env` (optional but recommended)

Create a `.env` file in project root and add:

```env
NEMOTRON_API_KEY=sk-or-v1-...
GLM_API_KEY=sk-or-v1-...
MINIMAX_API_KEY=sk-or-v1-...
```

If one or more keys are missing, the app runs in demo/fallback mode.

Optional runtime controls:

```env
# Weighted engine behavior
DYNAMIC_WEIGHT_USE_NEURAL=false

# Trace recording
RL_TRACE_ENABLED=true
RL_TRACE_PATH=logs/rl_trace.jsonl
RL_TRACE_MAX_BUFFER=2048
RL_TRACE_FLUSH_EVERY=32

# Training scaffold
GRPO_TRAIN_ENABLED=false
GRPO_TRAIN_EVERY_STEPS=16
GRPO_MIN_TRACE_BEFORE_TRAIN=256
REWARD_FLOOR=0.0
UNCERTAINTY_PENALTY_ENABLED=false
UNCERTAINTY_PENALTY_WEIGHT=0.0
REWARD_MODE=non_negative_v1
```

## Run Locally

### Start app (UI + API)

```bash
python app.py
```

You should see:
- FastAPI on `http://127.0.0.1:8000`
- Gradio on `http://127.0.0.1:7860`

### Run inference benchmark

```bash
python inference.py
```

Runs 5 episodes each for `easy`, `medium`, and `hard`, then prints total reward and normalized score.

### Run offline verification

```bash
python test_offline.py
```

Verifies data generation, sub-agents, council vote path, dissent logging, reward shaping, and state vector build.

## FastAPI Endpoints

- `POST /reset` - reset environment and return fresh observation.
- `POST /step` - apply action and return transition payload.
  - request body: `{"action": "APPROVE"}` (or `REJECT`, `REQUEST_MORE_INFO`)
- `GET /state` - return current observation.
- `GET /health` - basic health + episode + demo mode flag.

## Notes

- Default environment in `app.py` starts in `hard` task mode.
- Runtime artifacts are ignored by git (`logs/*.json`, `brain/*.pt`, `.env`).
- For implementation details and maintenance notes, see `docs/IMPLEMENTATION_GUIDE.md`.
