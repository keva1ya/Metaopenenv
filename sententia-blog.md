# Sententia
### *Verdict. From Latin. The considered judgment of a council.*

> Built at the **Meta PyTorch OpenEnv Hackathon × Scaler School of Technology** — India's first OpenEnv AI Hackathon, in collaboration with Meta, Hugging Face, and PyTorch. April 25–26, 2026. Bangalore.

---

## The Loan That Almost Slipped Through

It's 11:42 PM at a mid-sized bank.

A loan application for ₹4,80,000 lands in the queue. Credit score: 718 — solid. Employment: eight years at the same firm. Income: ₹90,000 declared annually.

Looks clean. But it isn't.

Spending volatility: **0.81**. Application consistency: **0.27**. KYC status: **PENDING**. And the loan amount? **5.3× the declared annual income.**

A human underwriter might catch it. Or might not — it's late, the credit score looked fine, and the queue has forty more behind it.

This is not a hypothetical. This is the daily operating reality of credit underwriting at scale. And it is exactly the problem **Sententia** was built to solve.

---

## What Is Wrong with Automated Lending Today

Most production loan systems are single-model pipelines. Input goes in. A score comes out. APPROVE or REJECT fires. There is no internal deliberation, no domain separation, no mechanism for the system to say *I'm not confident — give me more.*

When a single model is wrong, it is wrong completely and silently.

The deeper problem is architectural. Credit risk is not one problem. It is at least three distinct problems — financial capacity, behavioral integrity, and policy compliance — that require fundamentally different reasoning. Collapsing them into one scoring function means each domain's signal is diluted by the others. A fraud pattern gets averaged away by a good credit score. A policy violation gets obscured by strong income.

Single-model pipelines don't just miss edge cases. They have no way to *know* they're missing them.

---

## Sententia: A Full Credit Committee, Automated

Sententia is a **production-grade, multi-agent loan underwriting system** built on Meta's OpenEnv framework. It does not score loan applications. It *deliberates* on them — the way a real credit committee does — across three cascading levels of specialised intelligence.

```
Applicant Profile (18 fields)
        │
        ▼
┌──────────────────────────────────┐
│  LEVEL 1: SPECIALIST SUB-AGENTS  │  ← 9 deterministic domain experts
│  (Zero LLM cost · Pure logic)    │     Risk · Fraud · Policy desks
└────────────┬─────────────────────┘
             │ 9 structured reports
             ▼
┌──────────────────────────────────┐
│  LEVEL 2: COUNCIL HEAD AGENTS    │  ← 3 fine-tuned LLMs on Hugging Face
│  Qwen 2.5 7B · Mistral 7B ·     │     Domain-separated ensemble
│  Qwen 2.5 3B                     │     Hard rule fallback on failure
└────────────┬─────────────────────┘
             │ 3 votes + confidence + reasoning
             ▼
┌──────────────────────────────────┐
│  LEVEL 3: DYNAMIC WEIGHT ENGINE  │  ← Portfolio-aware final decision
│  (RL-traceable · GRPO-ready)     │     Override detection + dissent log
└──────────────────────────────────┘
             │
             ▼
      APPROVE · REJECT · REQUEST_MORE_INFO
```

---

## Level 1 — The Specialists

Nine sub-agents evaluate every application. Each focuses on exactly one thing. None of them call an LLM — they are fast, deterministic, and zero-cost. They convert raw applicant data into structured signal that the council can reason over.

The 18 applicant parameters are **strictly separated by agent domain** — a design decision that mirrors how real credit institutions compartmentalise information access.

### Financial Parameters → Risk & Policy Desks

| Parameter | Range | What It Signals |
|---|---|---|
| Income | ₹30K–₹200K/yr | Repayment capacity baseline |
| Credit Score | 300–850 | Bureau-reported creditworthiness |
| Employment Years | 0–40 | Income stability |
| Loan Amount | ₹50K–₹500K | Absolute exposure |
| Existing Debt | ₹0–₹200K | Current leverage pressure |
| Loan Term | 5–30 years | Duration risk |
| Monthly Expenses | ₹5K–₹80K | Disposable income signal |
| Previous Defaults | 0–5 | Historical repayment failure |
| GST Registered | Boolean | Business legitimacy |
| ITR Filed Years | 0–20 | Tax compliance history |
| Mortgage Value | ₹0–₹700K | Available collateral |
| Age, Dependents, Customer Score | — | Eligibility & loyalty signals |

### Behavioral Parameters → Fraud Desk Only

| Parameter | Range | What It Signals |
|---|---|---|
| Spending Volatility | 0.0–1.0 | Stress-application indicator |
| Loan Purpose Clarity | 0.0–1.0 | Intent transparency |
| Application Consistency | 0.0–1.0 | Cross-field contradiction detection |
| Bank KYC Status | VERIFIED / PENDING / FAILED | Identity verification state |

**The Fraud agent never sees income, credit score, or collateral.** The Risk agent never sees spending volatility or KYC status. This separation is enforced in the codebase — because in a real bank, these teams don't share raw data either, and that boundary is what makes each team's judgment independently credible.

### The Three Desks

**Risk Desk** evaluates creditworthiness through three agents: `CreditAnalystAgent` (bureau score, repayment history, utilization), `IncomeVerifierAgent` (income sufficiency, ITR compliance, GST registration), and `DebtAnalyzerAgent` (debt load, debt-to-income pressure).

**Fraud Desk** evaluates behavioral integrity: `PatternCheckerAgent` (spending volatility vs. loan size), `ConsistencyCheckerAgent` (cross-field contradiction detection), and `IdentityVerifierAgent` (KYC status, default history, purpose clarity).

**Policy Desk** evaluates bank compliance: `ComplianceOfficerAgent` (age 21–60, age+term ≤ 70), `TermValidatorAgent` (loan term within policy window), and `LTVCheckerAgent` (loan-to-value vs. collateral coverage).

Each agent returns three things: a **recommendation**, a **confidence score**, and **key flags** — a structured list of what triggered its decision. These flow upward as typed, validated reports.

---

## Level 2 — The Council

Three head agents receive the sub-agent reports for their domain. Each is powered by its own **fine-tuned, domain-specialised LLM** — together forming an ensemble where each model is the expert for its slice of underwriting.

| Head Agent | Model | Domain Specialisation |
|---|---|---|
| `RiskAgent` | **Qwen 2.5 7B** | Financial capacity to repay — tabular reasoning over credit, income, and employment signals |
| `PolicyAgent` | **Mistral 7B** | Bank policy & regulatory compliance — multi-rule evaluation over term, age, and LTV constraints |
| `FraudAgent` | **Qwen 2.5 3B** | Behavioral integrity & fraud signals — instruction-following over KYC, consistency, and spending patterns |

All three models are fine-tuned and hosted on Hugging Face. If any endpoint fails — network error, rate limit, model timeout — the agent falls back to direct rule evaluation in pure Python. **The system never stalls on a flaky endpoint.**

Each head agent synthesizes only the sub-reports relevant to its domain and returns a structured verdict:

```
DECISION: APPROVE | REJECT | REQUEST_MORE_INFO
CONFIDENCE: 0.0–1.0
REASONING: Two-sentence justification
```

### Dissent — The Feature Most Systems Skip

When the three council agents don't agree, Sententia does something most automated systems architecturally cannot: it **detects dissent, identifies the minority voter, captures their reasoning, and logs it permanently.**

```python
# From voting.py
if v.decision.value != majority_action:
    dissent_entries.append(DissentLog(
        dissenting_agent=v.agent_name,
        minority_decision=v.decision.value,
        majority_decision=majority_action,
        dissent_reason=v.reasoning,
        timestamp=datetime.utcnow().isoformat()
    ))
```

Every disagreement is recorded in `logs/dissent_log.json` and surfaced prominently in the UI.

This matters because a unanimous REJECT is routine. A **2-1 REJECT where the Fraud agent voted APPROVE** is a signal that deserves scrutiny — and in a real credit committee, that minority opinion is often the early warning the majority missed.

**Tie-breaking rule:** In a three-way split, the `PolicyAgent` casts the deciding vote — because a policy violation is a hard no regardless of financial capacity or behavioral signals.

---

## Level 3 — The Weighted Decision Engine

The council's output is a **recommendation**, not a final decision. The `DynamicWeightDecisionEngine` makes the final call.

It takes in:
- The three council votes and their confidence scores
- Full portfolio state: approved count, rejection rate, average reward over the last 10 decisions
- Manual multipliers set live from the UI — weight Risk higher for conservative portfolios, Fraud higher in high-volatility periods

It outputs a final action, normalized agent weights used in this decision, and an **override flag** if it disagreed with the council's recommendation.

The engine also feeds a **GRPO-compatible trace buffer** — every decision is recorded as a training trace with reward components, confidence scores, and portfolio diagnostics. This is the foundation for reinforcement learning fine-tuning of the weight network over time.

---

## The Reward System — Structurally Fraud-Proof

The reward signal is designed so no single metric can be gamed:

```
Reward = 0.4 × Financial Reward
       + 0.4 × Fraud Reward
       + 0.2 × Policy Reward
```

**Hard overrides fire before the reward function runs.** If strong fraud signals are present and the action is APPROVE, the reward is zero — regardless of the financial or policy score. If financial risk is critical, the same rule applies.

Reward hacking is structurally prevented, not policy-prevented. The RL layer cannot learn to approve known fraud cases because the financial reward happened to be high.

| Reward Component | Signals |
|---|---|
| Financial | Credit score · Income · Employment · Collateral coverage |
| Fraud | Spending volatility · Loan-to-income ratio · Employment tenure |
| Policy | Age limits · Loan term · Age+term ceiling · Debt ratio · Loan ceiling |

Mortgage mode is a special fast path: if collateral ≥ 1.1× loan amount, the application auto-approves with a reward of 1.0.

---

## Why Three Different LLMs?

The question judges will ask: why not one strong general-purpose model?

**Because domain specialisation removes systematic blind spots.**

A general-purpose LLM asked to assess both fraud signals and policy compliance carries the biases of both training objectives simultaneously. When it encounters a high-income applicant with suspicious spending patterns, it has no internal mechanism to treat those as separate concerns — they get averaged together in the same forward pass.

Sententia's architecture forces these concerns apart:
- Qwen 2.5 7B is trained to reason over tabular financial data — credit scores, income ratios, debt loads. That is all it sees.
- Mistral 7B is trained on multi-rule compliance evaluation. It has never seen a spending volatility score.
- Qwen 2.5 3B is trained specifically on behavioral signals and KYC patterns. It does not know the applicant's credit score.

Each model's judgment is **independently credible** because its training domain and its inference context are aligned. The ensemble's output is not three votes from generalists — it is one vote each from three specialists.

---

## The Interface — A Risk Desk, Not a Dashboard

The Gradio UI is built to feel like an actual risk workstation.

When you click **Next Applicant**, the interface enters a full skeleton processing state — every panel fills with live scanning indicators while the nine sub-agents run in parallel, the three council LLMs deliberate, and the weighted engine computes. Results replace the skeleton atomically when complete.

**What judges and operators can see and control:**
- Full financial profile and behavioral signal breakdown per applicant
- Per-domain sub-agent accordion reports with flags and confidence scores
- Council vote panel: each agent's verdict, assigned weight, and full two-sentence reasoning
- Dissent badge with dissenting agent identified and minority reasoning captured
- Weighted final decision with override detection highlighted
- Live dissent log showing the last 10 disagreements
- Portfolio health dashboard: approved / rejected / reviewed counts and average reward
- Runtime trace counter and GRPO training status
- Adjustable per-agent scoring multipliers (must sum to 100)
- Manual action override: force APPROVE / REJECT / REQUEST_MORE_INFO or defer to the engine

A **FastAPI backend** runs alongside at port 8000, exposing REST endpoints for programmatic access — `/reset`, `/step`, `/state`, and `/health`.

---

## Three Task Levels — From Heuristic to Full Architecture

| Level | Behavior |
|---|---|
| `easy` | Collateral heuristic only. If mortgage ≥ 1.1× loan → APPROVE. |
| `medium` | Risk score composite from credit, income, employment, and collateral. |
| `hard` | Full 3-level pipeline — all 9 sub-agents, council vote, weighted engine. |

The default runtime uses `hard` mode. The inference benchmark runs all three levels with five episodes each and returns a normalized score between 0.01 and 0.99.

---

## What This Is Not

This is not a chatbot.

It is not a toy demo with a loan-themed prompt wrapped around a general-purpose LLM.

It is not a single scoring function with a threshold.

**Sententia is a structured multi-agent decision environment** built on Meta's OpenEnv framework with:

- Typed, validated, immutable applicant observations (18 fields, Pydantic-enforced)
- Domain-separated agent responsibilities enforced at the codebase level
- Fine-tuned LLM ensemble with guaranteed deterministic fallback
- Democratic council voting with dissent capture and permanent logging
- Portfolio-aware final decision engine
- RL-compatible GRPO trace logging for offline fine-tuning
- A live interactive risk desk UI and a REST API backend

---

## The Bigger Picture

Global non-performing loan ratios have climbed steadily across emerging markets. The gap is not a shortage of data — applicants generate behavioral and financial signals continuously. The gap is a shortage of systems sophisticated enough to reason over those signals without collapsing their complexity into a single score.

Sententia is a proof of concept for a different architecture: one where the decision process is as auditable as the decision itself, where dissent is a first-class output, and where the RL feedback loop continuously improves the weight given to each domain of expertise.

The loan at 11:42 PM with the 0.27 consistency score?

Sententia catches it. It documents the catch. And it explains, agent by agent, exactly why.

---

*Built in 48 hours at the Meta PyTorch OpenEnv Hackathon × Scaler School of Technology, Bangalore, April 2026.*

*Stack: Python · Gradio · FastAPI · Pydantic · PyTorch · Hugging Face Inference Endpoints · OpenEnv · Qwen 2.5 7B · Mistral 7B · Qwen 2.5 3B*
