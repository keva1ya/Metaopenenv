---
title: AI Loan Approval OpenEnv
emoji: 🏦
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# The Loan That Almost Slipped Through

Picture this: it's 11:42 PM at a mid-sized bank. A loan application for ₹4,80,000 comes in. The applicant's credit score looks fine — 718. Income is declared at ₹90,000 a year. Employment is listed as eight years at the same firm.

But something is off.

The spending volatility score is 0.81. The application consistency is 0.27. The KYC status is PENDING. And the loan amount is 5.3× the declared annual income.

A human underwriter might catch it. Or might not — it's late, they're tired, and the credit score looked fine.

**This is exactly the problem we built to solve.**

---

# AI Loan Approval OpenEnv

> A production-grade, multi-agent loan underwriting system where specialized AI agents argue, dissent, vote, and finally render a weighted decision — just like a real credit committee, but faster, traceable, and never tired.

---

## The Problem with How Loans Are Approved Today

Most automated loan systems are single-model pipelines. They take inputs, run a scoring function, and spit out APPROVE or REJECT. There's no internal debate. No fraud vigilance independent from credit scoring. No policy officer asking "but does this even comply with age-plus-term rules?" No mechanism for the system to say "I'm not sure — show me more."

When a single model is wrong, it is wrong completely and silently.

We wanted to build something different.

---

## The Architecture: A Full Credit Committee, Automated

The system is organized as three cascading levels — each with a distinct responsibility, each acting as a check on the others.

```
Applicant Profile (18 fields)
        │
        ▼
┌─────────────────────────────────┐
│  LEVEL 1: SPECIALIST SUB-AGENTS │  ← 9 deterministic experts
│  (Zero LLM cost, pure logic)    │
└────────────┬────────────────────┘
             │ 9 reports (rec + confidence + flags)
             ▼
┌─────────────────────────────────┐
│  LEVEL 2: COUNCIL HEAD AGENTS   │  ← 3 LLM-powered synthesizers
│  (OpenRouter: 3 free models)    │  ← Hard rule fallback on failure
└────────────┬────────────────────┘
             │ 3 votes + reasoning + confidence
             ▼
┌─────────────────────────────────┐
│  LEVEL 3: DYNAMIC WEIGHT ENGINE │  ← Final decision
│  (Portfolio-aware, RL-traceable)│  ← Detects council override
└─────────────────────────────────┘
             │
             ▼
      APPROVE / REJECT / REQUEST_MORE_INFO
```

---

## Level 1 — The Specialists

Nine sub-agents evaluate every application. Each one focuses on exactly one thing. None of them talk to an LLM — they are fast, deterministic, and zero-cost.

### Risk Desk (3 agents)

| Agent | What It Checks |
|---|---|
| `CreditAnalystAgent` | Bureau score, repayment history, credit utilization |
| `IncomeVerifierAgent` | Income sufficiency, ITR filing years, GST registration |
| `DebtAnalyzerAgent` | Existing debt load, debt-to-income ratio pressure |

### Fraud Desk (3 agents)

| Agent | What It Checks |
|---|---|
| `PatternCheckerAgent` | Spending volatility vs. loan size — is this a stress application? |
| `ConsistencyCheckerAgent` | Do the application fields contradict each other? |
| `IdentityVerifierAgent` | KYC status, previous defaults, purpose clarity |

### Policy Desk (3 agents)

| Agent | What It Checks |
|---|---|
| `ComplianceOfficerAgent` | Age 21–60, age+term ≤ 70, term ≤ 25 years |
| `TermValidatorAgent` | Loan term within bank policy window |
| `LTVCheckerAgent` | Loan-to-value ratio vs. collateral coverage |

Each agent returns three things: a **recommendation**, a **confidence score**, and **key flags** — a short list of what triggered its decision. These flow upward as structured reports.

---

## Level 2 — The Council

Three head agents receive the sub-agent reports for their domain. Each is powered by a different free OpenRouter model — they literally think differently.

| Head Agent | Model | Domain |
|---|---|---|
| `RiskAgent` | Nemotron (nvidia/nemotron-4-340b) | Financial capacity to repay |
| `FraudAgent` | GLM (glm-z1-flash) | Behavioral integrity and fraud signals |
| `PolicyAgent` | MiniMax (minimax-m2.5) | Bank policy hard-rule compliance |

Each head agent receives only the sub-reports relevant to its domain, synthesizes them, and produces a structured verdict:

```
DECISION: APPROVE | REJECT | REQUEST_MORE_INFO
CONFIDENCE: 0.0–1.0
REASONING: Two-sentence justification
```

If an LLM call fails for any reason — network error, rate limit, model timeout — the agent **falls back to direct rule evaluation** in pure Python. The system never stalls.

### Dissent: When Agents Disagree

This is the part most automated systems skip entirely.

When the three council agents don't agree, the system **detects dissent, identifies the minority voter, captures their reasoning, and logs it** with a timestamp. The UI surfaces this prominently. Every disagreement is permanently recorded in `logs/dissent_log.json`.

This matters because real credit committees also disagree — and those minority opinions are often the early signal of a risk that the majority missed.

```python
# From voting.py — the council never silences a minority view
if v.decision.value != majority_action:
    dissent_entries.append(DissentLog(
        dissenting_agent=v.agent_name,
        minority_decision=v.decision.value,
        majority_decision=majority_action,
        dissent_reason=v.reasoning,
        ...
    ))
```

**Tie-breaking rule:** In a three-way split, the `PolicyAgent` casts the deciding vote — because a policy violation is a hard no regardless of financial capacity.

---

## Level 3 — The Weighted Decision Engine

The council's output is a **recommendation**, not a final decision. The `DynamicWeightDecisionEngine` makes the final call.

It takes in:

- The three council votes and their confidence scores
- The full portfolio state (approved count, rejection rate, average reward over last 10 decisions)
- Manual multipliers set live from the UI (e.g., weight Risk higher for conservative portfolios)

It outputs:
- A final action: `APPROVE`, `REJECT`, or `REQUEST_MORE_INFO`
- Normalized agent weights used in this decision
- An override flag if it disagreed with the council's recommendation

The engine also feeds into a **GRPO-compatible trace buffer** — every decision is recorded as a training trace with reward components, confidence scores, and portfolio diagnostics. This is the foundation for reinforcement learning fine-tuning of the weight network.

---

## The Reward System

The reward signal is designed so no single metric can be gamed.

```
Reward = 0.4 × Financial Reward
       + 0.4 × Fraud Reward
       + 0.2 × Policy Reward
```

**Hard overrides come first.** If strong fraud signals are present and the action is APPROVE, the reward is zero — regardless of the financial or policy score. If financial risk is critical, same rule applies. The system punishes reward hacking before it can start.

| Component | Signal |
|---|---|
| Financial | Credit score, income, employment, collateral coverage |
| Fraud | Spending volatility, loan-to-income ratio, employment tenure |
| Policy | 5 hard bank rules: age, term, age+term, debt ratio, loan ceiling |

Mortgage mode is a special fast path: if collateral ≥ 1.1× loan amount, the application auto-approves and reward is 1.0.

---

## The Applicant Profile

Every synthetic applicant carries 18 fields — a mix of financial signals and behavioral markers that intentionally belong to different agents.

**Financial (used by Risk + Policy):**
Income · Credit Score · Employment Years · Loan Amount · Loan Type · Mortgage Value · Loan Term · Existing Debt · Age · Monthly Expenses · Dependents · Previous Defaults · GST Registered · ITR Filed Years · Customer Score

**Behavioral (used by Fraud only):**
Spending Volatility · Loan Purpose Clarity · Application Consistency · Bank KYC Status

This separation is enforced in the codebase — the Fraud head agent receives behavioral context only. The Risk and Policy agents receive financial context only. No agent gets to see the full picture by design, because in a real bank, neither do they.

---

## The Interface

The Gradio UI is built to feel like a risk desk, not a dashboard.

When you click **Next Applicant**, the entire interface enters a skeleton processing state — all panels fill with live scanning indicators while the 9 sub-agents run, the 3 council LLMs deliberate, and the weighted engine computes. When results arrive, the full profile replaces the skeleton atomically.

**What you can see and control:**

- Full financial profile table per applicant
- Behavioral signal breakdown
- Per-domain sub-agent reports (Risk / Fraud / Policy accordions)
- Council vote panel — each agent's verdict, weight, and full reasoning
- Dissent badge with dissenting agent identified
- Weighted final decision with override detection
- Live dissent log (last 10 entries)
- Portfolio health dashboard (approved / rejected / reviewed / avg reward)
- Runtime trace counter and GRPO training status
- Adjustable per-agent scoring multipliers (must sum to 100)
- Action override: force APPROVE / REJECT / REQUEST_MORE_INFO or let the engine decide

**FastAPI backend** runs alongside at port 8000, exposing REST endpoints for programmatic access.

---

## Task Levels

| Level | Behavior |
|---|---|
| `easy` | Collateral heuristic only. If mortgage ≥ 1.1× loan → APPROVE. |
| `medium` | Risk score composite from credit, income, employment, collateral. |
| `hard` | Full 3-level pipeline — all 9 sub-agents, council vote, weighted engine. |

The default runtime uses `hard` mode. The inference script (`scripts/inference.py`) benchmarks all three levels with 5 episodes each and returns a normalized score between 0.01 and 0.99.

---

## Repository Layout

```
Metaopenenv/
├── app.py                          ← Gradio UI + FastAPI backend
├── openenv.yaml                    ← Environment config
├── requirements.txt
├── agents/
│   ├── council/
│   │   ├── risk_agent.py           ← RiskAgent (Nemotron)
│   │   ├── fraud_agent.py          ← FraudAgent (GLM)
│   │   └── policy_agent.py         ← PolicyAgent (MiniMax)
│   ├── sub_agents/
│   │   ├── risk/                   ← CreditAnalyst, IncomeVerifier, DebtAnalyzer
│   │   ├── fraud/                  ← PatternChecker, ConsistencyChecker, IdentityVerifier
│   │   └── policy/                 ← ComplianceOfficer, TermValidator, LTVChecker
│   └── voting.py                   ← Majority vote + dissent detection + logging
├── brain/
│   ├── dynamic_weight_net.py       ← Final weighted decision engine
│   ├── reward_shaper.py            ← 3-component reward with hard overrides
│   ├── state_builder.py            ← 28-dim state vector for RL
│   ├── trace_buffer.py             ← GRPO-compatible trace storage
│   └── grpo_trainer.py             ← Training scaffold
├── env/
│   ├── observation.py              ← Pydantic applicant model (18 fields, frozen)
│   ├── action.py                   ← APPROVE / REJECT / REQUEST_MORE_INFO
│   ├── loan_env.py                 ← OpenEnv-style environment
│   └── data_generator.py           ← Synthetic applicant generator
├── scripts/
│   ├── inference.py                ← Multi-level benchmark
│   └── test_offline.py             ← Full architecture verification (no PyTorch)
├── logs/                           ← Dissent log, RL trace (gitignored contents)
└── docs/
    └── IMPLEMENTATION_GUIDE.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```env
NEMOTRON_API_KEY=sk-or-v1-...
GLM_API_KEY=sk-or-v1-...
MINIMAX_API_KEY=sk-or-v1-...
```

All three keys are **free OpenRouter models**. If any key is missing, all head agents run in deterministic fallback mode — the system never breaks.

**Optional runtime controls:**

```env
DYNAMIC_WEIGHT_USE_NEURAL=false      # neural vs. linear weight engine
RL_TRACE_ENABLED=true                # record decisions for RL training
RL_TRACE_PATH=logs/rl_trace.jsonl
GRPO_TRAIN_ENABLED=false             # run GRPO training scaffold
GRPO_TRAIN_EVERY_STEPS=16
REWARD_FLOOR=0.0
UNCERTAINTY_PENALTY_ENABLED=false
REWARD_MODE=non_negative_v1
```

### 3. Run the app

```bash
python app.py
```

Opens:
- Gradio UI → `http://127.0.0.1:7860`
- FastAPI API → `http://127.0.0.1:8000`

### 4. Verify the architecture (no GPU, no API keys needed)

```bash
python scripts/test_offline.py
```

Runs all 9 sub-agents, council vote, dissent detection, reward shaper, and state vector builder. Verifies every architectural component end-to-end.

### 5. Run the inference benchmark

```bash
python scripts/inference.py
```

Benchmarks all three task levels (5 episodes each). Prints per-episode reward and a normalized final score.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Reset environment, get fresh applicant observation |
| `/step` | POST | Apply action → `{"action": "APPROVE"}` |
| `/state` | GET | Return current observation |
| `/health` | GET | Health check, episode count, demo mode flag |

---

## Design Decisions Worth Noting

**Why three different LLMs?**
Each head agent uses a different model. Risk uses Nemotron (strong at financial reasoning). Fraud uses GLM (strong at behavioral pattern recognition). Policy uses MiniMax (strong at structured rule application). Diversity in the council reduces systematic blind spots.

**Why deterministic sub-agents?**
LLM calls are expensive and slow. The nine Level-1 specialists are pure Python — they run in milliseconds, have zero cost, and produce perfectly traceable outputs. The LLMs get a pre-digested signal packet, not raw numbers.

**Why dissent logging?**
Because the most interesting thing in any credit decision is not the verdict — it's who objected, and why. A unanimous REJECT is routine. A 2-1 REJECT where the Fraud agent voted APPROVE is a signal that deserves scrutiny.

**Why REQUEST_MORE_INFO?**
Real credit committees don't always binary vote. Sometimes the right answer is "we need the applicant's last three ITR statements before we can decide." The action space includes this explicitly.

**Why a reward system with hard overrides?**
So the RL layer can't learn to approve known fraud cases because the financial reward happens to be high. Hard overrides fire before the reward function runs and return zero. Reward hacking is structurally prevented.

---

## What This Is Not

This is not a chatbot. It is not a toy demo. It is not a wrapper around a single LLM with a loan-themed prompt.

It is a structured multi-agent decision environment with:
- Typed, validated, immutable observations
- Domain-separated agent responsibilities
- LLM synthesis with guaranteed fallback
- Democratic council voting with dissent capture
- Portfolio-aware final decision
- RL-compatible trace logging
- A live interactive risk desk UI

---

## Acknowledgements

Built on [OpenRouter](https://openrouter.ai) for free LLM access, [Gradio](https://gradio.app) for the UI, [FastAPI](https://fastapi.tiangolo.com) for the backend, [Pydantic](https://docs.pydantic.dev) for type-safe observations, and [PyTorch](https://pytorch.org) for the weight network.
