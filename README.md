---
title: Sententia MetaOpenEnv
emoji: ⚖️
colorFrom: red
colorTo: gray
sdk: docker
app_file: app.py
pinned: false
---

<div align="center">
  <img src="https://img.shields.io/badge/Status-Production_Ready-success?style=for-the-badge&logo=rocket" alt="Status"/>
  <img src="https://img.shields.io/badge/Architecture-Multi--Agent_Hierarchical-6B4226?style=for-the-badge" alt="Architecture"/>
  <img src="https://img.shields.io/badge/Models-Qwen_|_Mistral-092E20?style=for-the-badge" alt="Models"/>
  <img src="https://img.shields.io/badge/Framework-Gradio_|_PyTorch-FF7D00?style=for-the-badge" alt="Framework"/>
</div>

<br/>

<div align="center">
  <h1>🏦 Sententia (MetaOpenEnv): A Full Credit Committee, Automated</h1>
  <p><em>Production-grade, RL-ready multi-agent underwriting built on Meta's OpenEnv framework.</em></p>
</div>

---

## 💡 The Problem: What Is Wrong with Automated Lending Today

Most production loan systems are single-model pipelines. Input goes in. A score comes out. `APPROVE` or `REJECT` fires. There is no internal deliberation, no domain separation, and no mechanism for the system to say *"I'm not confident — give me more."*

Credit risk is not one problem. It is three distinct problems: **financial capacity, behavioral integrity, and policy compliance**. Collapsing them into a single scoring function dilutes the signal. A fraud pattern gets averaged away by a good credit score; a policy violation is obscured by strong income.

Single-model pipelines don’t just miss edge cases. They have no way to *know* they’re missing them.

## 🚀 The Solution: Sententia

Sententia does not *score* loan applications. It *deliberates* on them across three cascading levels of specialized intelligence, mirroring the structure of a real-world credit committee.

```mermaid
flowchart TD
    subgraph External["External Interaction"]
        User["User / Evaluator"]
        REST["FastAPI REST Clients"]
    end

    subgraph Runtime["Production Runtime (Hugging Face Space)"]
        direction TB
        
        subgraph Security["Security Layer"]
            Auth["API Key / UI Basic Auth"]:::red
            DissentSanitizer["Dissent Sanitizer"]:::gray
            RateLimiter["Rate Limiter (slowapi)"]
        end

        subgraph L3["Level 3: Orchestration & Aggregation"]
            Aggregator["Deterministic Aggregator<br/>(Confidence × Manual Multipliers)"]:::orange
            AppPy["app.py (Gradio + FastAPI)"]
            OpenEnv["loan_env.py (OpenEnv)"]
            RewardShaper["RewardShaper<br/>0.4×Financial + 0.4×Fraud + 0.2×Policy<br/>⚠️ Hard Override → 0.0 if strong fraud/risk"]:::orange
            RewardScore["Reward Score [0.0 – 1.0]"]:::yellow
            UIDisplay["Gradio UI Score Display"]
            TraceLog[("logs/rl_trace.jsonl<br/>obs · action · reward · votes · weights")]:::blue
        end

        subgraph L1["Level 1: Deterministic Sub-Agents"]
            direction LR
            R_L1["Risk Analysts (3)"]
            F_L1["Fraud Detectors (3)"]
            P_L1["Policy Officers (3)"]
        end
    end

    subgraph OfflineRL["Offline RL Lifecycle (GPU Rig)"]
        direction TB
        PrepDataset["prepare_grpo_dataset.py<br/>reward → per-head credit"]
        TrainGRPO["train_grpo_offline.py<br/>(TRL + GRPO)"]
    end

    subgraph L2["Level 2: Council (Hugging Face Inference API)"]
        direction LR
        FAgent["FraudAgent<br/>(qwen-2.5-fraud)"]:::green
        PAgent["PolicyAgent<br/>(mistral-risk-model)"]:::purple
        RAgent["RiskAgent<br/>(qwen-risk-merged)"]:::pink
    end

    %% Connections
    User --> Auth
    REST --> Auth
    Auth --> RateLimiter
    RateLimiter --> AppPy
    
    AppPy --> OpenEnv
    OpenEnv --> R_L1 & F_L1 & P_L1
    
    R_L1 -- "Sub-agent Reports" --> RAgent
    F_L1 -- "Sub-agent Reports" --> FAgent
    P_L1 -- "Sub-agent Reports" --> PAgent

    RAgent & FAgent & PAgent -- "Votes + Reasoning" --> Aggregator
    Aggregator -- "Final Action" --> AppPy
    
    OpenEnv -- "obs + final_action" --> RewardShaper
    RewardShaper --> RewardScore
    RewardScore --> UIDisplay
    RewardScore -- "stored in TraceRecord" --> TraceLog
    
    TraceLog -- "sync" --> PrepDataset
    PrepDataset -- "reward credited per head" --> TrainGRPO
    TrainGRPO -- "updated adapters" --> RAgent & FAgent & PAgent

    %% Class Definitions
    classDef red fill:#fce4e4,stroke:#f8b4b4,stroke-width:2px,color:#333
    classDef gray fill:#e5e7eb,stroke:#d1d5db,stroke-width:2px,color:#333
    classDef orange fill:#fed7aa,stroke:#fdba74,stroke-width:2px,color:#333
    classDef yellow fill:#fef08a,stroke:#fde047,stroke-width:2px,color:#333
    classDef blue fill:#bfdbfe,stroke:#93c5fd,stroke-width:2px,color:#333
    classDef green fill:#bbf7d0,stroke:#86efac,stroke-width:2px,color:#333
    classDef purple fill:#e9d5ff,stroke:#d8b4fe,stroke-width:2px,color:#333
    classDef pink fill:#fbcfe8,stroke:#f9a8d4,stroke-width:2px,color:#333
```

---

## ⚙️ Technical Deep Dive

### 1. Data Ingestion & Pydantic Validation
The environment natively accepts data via CSV/JSON bulk uploads or manual UI entry. Raw data is validated using a Pydantic `LoanObservation` class. The system supports **unlimited input ranges** for financial fields (Loan Term, Monthly Expenses, etc.), allowing for advanced stress-testing beyond traditional banking boundaries.

### 2. Level 1: Specialized Sub-Agents (Rule-Based + LLM Signal)
Nine sub-agents evaluate the application within isolated domains. They use a hybrid of deterministic rules and lightweight LLM signals for fluid policy checks.
*   **Risk Desk:**
    *   `CreditAnalystAgent`: Assesses bureau scores and repayment history.
    *   `IncomeVerifierAgent`: Validates income sufficiency and DTI.
    *   `DebtAnalyzerAgent`: Examines current leverage and total debt limits.
*   **Fraud Desk:**
    *   `PatternCheckerAgent`: Evaluates spending volatility vs. loan size.
    *   `ConsistencyCheckerAgent`: Scans for cross-field contradictions.
    *   `IdentityVerifierAgent`: Evaluates Bank KYC status (VERIFIED/PENDING/FAILED).
*   **Policy Desk:**
    *   `ComplianceOfficerAgent`: Verifies applicant is within the 21–60 age bracket.
    *   `TermValidatorAgent`: Ensures loan terms fit institutional bounds.
    *   `LTVCheckerAgent`: Ensures Loan-to-Value (collateral vs. amount) is mathematically sound.

### 3. Level 2: The Council (Domain-Separated LLM Ensemble)
Three head agents synthesize the 9 sub-reports. Each is powered by a **fine-tuned, domain-specialized LLM** accessed via `huggingface_hub.InferenceClient`.
*   **`RiskAgent` (`Keva1ya/qwen-risk-merged`):** Tabular reasoning over financial capacity.
*   **`FraudAgent` (`Keva1ya/qwen-2.5-fraud`):** Behavioral integrity. Explicitly enforces a hard `REJECT` veto on failed KYC checks.
*   **`PolicyAgent` (`Keva1ya/mistral-risk-model`):** Multi-rule compliance evaluation using Mistral's instruction templates.
*   *Fail-Safe:* If the Hugging Face API is unavailable, agents gracefully default to rule-based fallback logic.

### 4. Level 3: Dynamic Weight Decision Engine & Dissent Logging
The council submits votes (`APPROVE`, `REJECT`, `REQUEST_MORE_INFO`) to the `VotingEngine`.
*   **Dissent Logging:** If the council vote is 2-1, the engine detects dissent, identifies the minority voter, captures their specific two-sentence reasoning, and saves it permanently to `logs/dissent_log.json`.
*   **Tie-Breaking:** The `PolicyAgent` functions as the ultimate tie-breaker for strict regulatory compliance.
*   **Dynamic Multipliers:** Evaluators can manually scale the mathematical weight of Risk, Fraud, or Policy via the UI.

### 5. Structurally Fraud-Proof Reward Shaper
The environment simulates feedback via a robust `RewardShaper`:
```python
Reward = (0.4 * Financial_Score) + (0.4 * Fraud_Score) + (0.2 * Policy_Score)
```
**Hard Overrides:** If the applicant presents severe fraud patterns or a `FAILED` KYC status, the reward is automatically crushed to `0.0`. The RL framework cannot learn to approve fraudulent loans simply because the financial reward is high.

### 6. The Big Brain Server (Offline GRPO RL)
The system is built for **Group Relative Policy Optimization (GRPO)**.
*   **State Trace Buffer:** Every observation, confidence score array, decision, and calculated reward is mathematically compressed into a `float32` state tensor (shape: `28,`) and saved to `logs/rl_trace.jsonl`.
*   **Offline GPU Rig Training:** The `scripts/train_grpo_offline.py` loop reads the trace buffer and utilizes PyTorch and Hugging Face `TRL` (Transformer Reinforcement Learning) to run backpropagation against the `dynamic_weight_net`. 
*   *Purpose:* The RL network does not learn how to underwrite; it learns **how much weight to assign to each Head Agent** based on historical reward trajectories.

### 7. Task Difficulties (Environment Constraints)
*   **`easy`**: Bypasses the LLM council entirely. Evaluates only collateral heuristics (Mortgage Mode Auto-Approval).
*   **`medium`**: Synthesizes a deterministic risk composite score.
*   **`hard`**: The default mode. Triggers the full 3-level pipeline, Hugging Face LLM endpoints, and Dissent Logging.

---

## 💻 Quick Start & Installation

Run the local Risk Desk (Gradio UI + FastAPI backend) to visually process single applicants or batch-process CSV datasets.

### Prerequisites
```bash
git clone https://github.com/keva1ya/Metaopenenv.git
cd Metaopenenv
pip install -r requirements.txt
```

### Configuration
Add your Hugging Face API Token to a `.env` file to enable the Level-2 LLM Head Agents:
```env
HF_TOKEN=hf_your_secure_token_here
```

### Launch
Start the complete Gradio interactive dashboard:
```bash
python3 app.py
```

### Offline Automated Benchmarking
Run the headless offline test benchmark to see the environment interact programmatically:
```bash
python3 scripts/test_offline.py
```

---
*Built for the Meta PyTorch OpenEnv Hackathon × Scaler School of Technology. India's first OpenEnv AI Hackathon.*
