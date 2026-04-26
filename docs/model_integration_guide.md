# Metaopenenv Model Integration Guide

This guide describes the **Ensemble Model Pipeline** that powers the three council head agents in Metaopenenv. Each head agent is wired to its own fine-tuned, domain-specialised LLM hosted on Hugging Face. The three models vote, dissent is logged, and the weighted decision engine produces the final loan action.

---

## 1. Agent → Model Mapping

| Council Head | Fine-Tuned Model | Trained On | Hosting |
| --- | --- | --- | --- |
| **RiskAgent** | **Qwen 2.5** (consumer & personal credit risk) | Google Colab | Hugging Face |
| **PolicyAgent** | **Mistral 7B** (bank policy & SBA-style compliance) | Local workstation | Hugging Face |
| **FraudAgent** | **Llama 3** (transaction & behavioural fraud) | Local workstation | Hugging Face |

> Together these three specialised LLMs form an **ensemble**: each is the expert for its own slice of the underwriting decision. The voting engine + dynamic weight engine combine their judgments into a single bank action.

### A. Qwen 2.5 — Risk Agent
- **Base:** `Qwen2.5-7B-Instruct`
- **Fine-tuned on:** Tabular consumer credit datasets (income, employment, DTI, credit score, defaults).
- **Why Risk:** Qwen 2.5's strong tabular reasoning makes it ideal for the **Chief Risk Officer** role, which evaluates *financial fields only* (no behavioural signals).
- **Training environment:** Google Colab (T4 / A100 with QLoRA 4-bit).
- **Upload target:** `your-username/qwen-credit-risk` on Hugging Face.

### B. Mistral 7B — Policy Agent
- **Base:** `Mistral-7B-Instruct`
- **Fine-tuned on:** SBA loan compliance data, term/age policy windows, LTV / mortgage coverage rules.
- **Why Policy:** Mistral 7B reasons well over multi-rule, multi-field policy windows and integrates 3 sub-agent reports (ComplianceOfficer, TermValidator, LTVChecker).
- **Training environment:** Local (RTX-class GPU with QLoRA 4-bit).
- **Upload target:** `your-username/mistral-policy-officer` on Hugging Face.

### C. Llama 3 — Fraud Agent
- **Base:** `Llama-3.1-8B-Instruct`
- **Fine-tuned on:** Transaction-level fraud labels (CASH_IN / CASH_OUT / TRANSFER patterns, balance inconsistencies).
- **Why Fraud:** Llama 3's strong instruction-following lets the **Chief Fraud Officer** combine PatternChecker, ConsistencyChecker, and KYCVerifier signals while honouring the hard KYC veto.
- **Training environment:** Local (Apple Silicon / CUDA with QLoRA 4-bit).
- **Upload target:** `your-username/llama3-fraud-detector` on Hugging Face.

---

## 2. The Ensemble Pipeline

```
                         ┌───────────────────────────┐
                         │     LoanObservation       │
                         │ (financial + behavioural) │
                         └──────────────┬────────────┘
                                        │
       ┌────────────────────────────────┼────────────────────────────────┐
       │                                │                                │
       ▼                                ▼                                ▼
┌────────────────┐             ┌────────────────┐              ┌────────────────┐
│  RiskAgent     │             │  PolicyAgent   │              │  FraudAgent    │
│  (Qwen 2.5)    │             │  (Mistral 7B)  │              │  (Llama 3)     │
│  HF endpoint   │             │  HF endpoint   │              │  HF endpoint   │
└──────┬─────────┘             └──────┬─────────┘              └──────┬─────────┘
       │ APPROVE/REJECT/RMI           │ APPROVE/REJECT/RMI            │ APPROVE/REJECT/RMI
       └──────────────┬───────────────┴────────────────┬──────────────┘
                      ▼                                ▼
              ┌──────────────────────────────────────────────┐
              │     VotingEngine + DissentLog                │
              └─────────────────────┬────────────────────────┘
                                    ▼
              ┌──────────────────────────────────────────────┐
              │  DynamicWeightDecisionEngine + RewardShaper  │
              │      → final action: APPROVE/REJECT/RMI      │
              └──────────────────────────────────────────────┘
```

Each head agent has a **deterministic fallback** that fires automatically if the model call errors, so the pipeline never blocks on a flaky inference endpoint.

---

## 3. Configuration: API Keys & Endpoints

The pipeline looks for credentials in the following resolution order **per agent**:

1. **Per-agent vars** (highest priority):
   `<AGENT>_API_KEY`, `<AGENT>_API_BASE`, `<AGENT>_MODEL`
2. **Shared HF token + per-agent model**:
   `HF_TOKEN` + `<AGENT>_MODEL`
   (uses the default Hugging Face Inference Providers router)
3. **Legacy OpenRouter fallback**:
   `NEMOTRON_API_KEY` (Risk) / `GLM_API_KEY` (Fraud) / `MINIMAX_API_KEY` (Policy), or `OPENROUTER_API_KEY`
4. **Demo mode**: deterministic rule-based fallback (no LLM)

Drop these into `.env` (placeholders already provided — just paste your keys):

```dotenv
# Shared token (used when per-agent *_API_KEY is unset)
HF_TOKEN=""

# RiskAgent → Qwen 2.5
RISK_MODEL=""        # e.g. "your-username/qwen-credit-risk"
RISK_API_KEY=""
RISK_API_BASE=""     # leave blank to use HF Inference Providers router

# PolicyAgent → Mistral 7B
POLICY_MODEL=""      # e.g. "your-username/mistral-policy-officer"
POLICY_API_KEY=""
POLICY_API_BASE=""

# FraudAgent → Llama 3
FRAUD_MODEL=""       # e.g. "your-username/llama3-fraud-detector"
FRAUD_API_KEY=""
FRAUD_API_BASE=""
```

> [!TIP]
> The default `<AGENT>_API_BASE` is `https://router.huggingface.co/v1` (HF Inference Providers, OpenAI-compatible). Override it when you deploy:
> - **HF Inference Endpoints** → `https://<endpoint>.endpoints.huggingface.cloud/v1/`
> - **Local vLLM / Ollama** → `http://localhost:8000/v1` (different port per model)

---

## 4. Connection Recipes

### Recipe A: All three on Hugging Face Inference Endpoints (production)

1. Push each merged checkpoint to Hugging Face:
   ```bash
   huggingface-cli upload your-username/qwen-credit-risk      ./qwen-merged
   huggingface-cli upload your-username/mistral-policy-officer ./mistral-merged
   huggingface-cli upload your-username/llama3-fraud-detector ./llama3-merged
   ```
2. Spin up an Inference Endpoint per repo on Hugging Face.
3. Drop the endpoint URLs into `.env`:
   ```dotenv
   HF_TOKEN="hf_..."
   RISK_MODEL="your-username/qwen-credit-risk"
   RISK_API_BASE="https://qwen-credit-risk.endpoints.huggingface.cloud/v1/"
   POLICY_MODEL="your-username/mistral-policy-officer"
   POLICY_API_BASE="https://mistral-policy-officer.endpoints.huggingface.cloud/v1/"
   FRAUD_MODEL="your-username/llama3-fraud-detector"
   FRAUD_API_BASE="https://llama3-fraud-detector.endpoints.huggingface.cloud/v1/"
   ```

### Recipe B: Local vLLM serving (development / no GPU bills)

Run each model on a separate port; the env wires them up automatically.

```bash
python -m vllm.entrypoints.openai.api_server --model your-username/qwen-credit-risk      --port 8001
python -m vllm.entrypoints.openai.api_server --model your-username/mistral-policy-officer --port 8002
python -m vllm.entrypoints.openai.api_server --model your-username/llama3-fraud-detector --port 8003
```

```dotenv
RISK_API_BASE="http://localhost:8001/v1"
RISK_API_KEY="local"
RISK_MODEL="your-username/qwen-credit-risk"

POLICY_API_BASE="http://localhost:8002/v1"
POLICY_API_KEY="local"
POLICY_MODEL="your-username/mistral-policy-officer"

FRAUD_API_BASE="http://localhost:8003/v1"
FRAUD_API_KEY="local"
FRAUD_MODEL="your-username/llama3-fraud-detector"
```

### Recipe C: Mixed (e.g. Qwen on HF, Mistral & Llama 3 local)

The per-agent vars are independent — set HF endpoint URL for Qwen and `localhost` URLs for Mistral / Llama 3 in the same `.env`.

---

## 5. Prompt Formats per Agent

Because each model was fine-tuned on a specific instruction format, the agents in `agents/council/*.py` already format inputs accordingly. The fine-tuning datasets should match these layouts for maximum accuracy.

### RiskAgent — Qwen 2.5 (consumer credit risk)
`agents/council/risk_agent.py` sends financial-only fields:
```python
user_prompt = f"""
Sub-agent reports:
{reports_str}

Financial context:
  Income: {obs.income:.0f}
  Credit Score: {obs.credit_score}
  Employment: {obs.employment_years}yr
  Loan Amount: {obs.loan_amount:.0f}
  Existing Debt: {obs.existing_debt:.0f}
  Mortgage Value: {obs.mortgage_value:.0f}
  Loan Term: {obs.loan_term}yr

Respond ONLY:
DECISION: <APPROVE|REJECT|REQUEST_MORE_INFO>
CONFIDENCE: <0.0-1.0>
REASONING: <2 sentences>
"""
```

### PolicyAgent — Mistral 7B (bank policy & SBA compliance)
`agents/council/policy_agent.py` sends the broader application context plus 3 policy sub-agent reports:
```python
user_prompt = f"""
Sub-agent reports:
{reports_str}

Application context:
  Age: {obs.age}
  Loan Term: {obs.loan_term}yr
  Loan Amount: {obs.loan_amount:.0f}
  Income: {obs.income:.0f}
  Existing Debt: {obs.existing_debt:.0f}
  Mortgage Value: {obs.mortgage_value:.0f}
  Loan Type: {obs.loan_type}
  ...

Respond ONLY:
DECISION: <APPROVE|REJECT|REQUEST_MORE_INFO>
CONFIDENCE: <0.0-1.0>
REASONING: <2 sentences>
"""
```

### FraudAgent — Llama 3 (behavioural fraud)
`agents/council/fraud_agent.py` sends behavioural fields plus identity / KYC guidance:
```python
user_prompt = f"""
Sub-agent reports:
{reports_str}

Behavioral context:
  Spending Volatility: {obs.spending_volatility:.2f}
  Loan Purpose Clarity: {obs.loan_purpose_clarity:.2f}
  Application Consistency: {obs.application_consistency:.2f}
  Previous Defaults: {obs.previous_defaults}

Identity/KYC guidance:
  Bank KYC status: {identity_status}
  Pattern fraud score: {fraud_score}
  Consistency flag count: {consistency_flags}

Respond ONLY:
DECISION: <APPROVE|REJECT|REQUEST_MORE_INFO>
CONFIDENCE: <0.0-1.0>
REASONING: <2 sentences>
"""
```

---

## 6. Validation Checklist

1. **Smoke-test each endpoint individually**: hit each `<AGENT>_API_BASE` with a trivial chat request to confirm auth and routing.
2. **Run the offline harness**:
   ```bash
   python scripts/test_offline.py
   ```
   This exercises sub-agents and grading without LLMs.
3. **Run the full pipeline**:
   ```bash
   python app.py
   ```
   Open the Gradio UI at `http://127.0.0.1:7860` — the `demo_mode` banner disappears once all three agents are configured.
4. **Compare to baseline**: log a batch of 100 applicants and compare ensemble outputs vs. the legacy OpenRouter agents to measure accuracy uplift.

> [!IMPORTANT]
> If you run all three models locally, three 7B / 8B models in 4-bit quantization need ~18-22 GB VRAM combined. Run them sequentially or split across machines if VRAM is tight.

---

## 7. Failure Modes & Safety Nets

- **Model down / 5xx**: `evaluate()` catches the exception and falls back to the agent's deterministic logic (majority sub-agent vote for Risk, soft-aggregation for Policy, identity-guarded fallback for Fraud).
- **Bad model output**: the response parser defaults to `REJECT` with `confidence=0.5` if it cannot extract a decision, so a malformed completion never silently approves a loan.
- **No keys at all**: the env logs `DEMO MODE: No HuggingFace or OpenRouter keys found` and the entire council runs on deterministic rules — useful for CI and offline demos.
