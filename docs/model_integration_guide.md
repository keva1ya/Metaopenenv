# Metaopenenv Model Integration Guide

This guide provides a comprehensive overview of the two models recently fine-tuned for the Metaopenenv multi-agent loan approval system: the **Mistral-7B SBA Risk Model** and the **Qwen2.5 Credit Risk Model**. 

It details their specific purposes, capabilities, and step-by-step instructions on how to connect and utilize them within the broader Metaopenenv ecosystem.

---

## 1. The Fine-Tuned Models: An Overview

We have successfully trained two specialized language models using QLoRA 4-bit quantization. These models are designed to replace generic LLMs (like standard GPT-4 or generic open-source models) with highly specialized financial evaluation engines.

### A. Mistral-7B SBA Risk Model
- **Base Model:** `Mistral-7B-Instruct`
- **Location:** `d:\projects\Metaopenenv\mistral-risk-merged` (Merged checkpoint) / `mistral-risk-lora-checkpoints` (Adapters)
- **Primary Focus:** **Small Business Administration (SBA) Loan Default Risk Assessment.**
- **Capabilities:** Trained specifically on SBA loan data. It understands business loan nuances, collateral requirements, GST registration impacts, and historical default patterns for small businesses.
- **Ideal Agent Assignment:** **Risk Agent (Head Agent)** or specialized business loan sub-agents.

### B. Qwen2.5 Credit Risk Model
- **Base Model:** `Qwen2.5-7B-Instruct`
- **Location:** `d:\projects\Metaopenenv\qwen_model`
- **Primary Focus:** **General Consumer & Personal Credit Risk Assessment.**
- **Capabilities:** Fine-tuned on standard tabular credit risk datasets. Highly adept at parsing individual consumer metrics like age, income, employment years, existing debt, and DTI (Debt-to-Income) ratios.
- **Ideal Agent Assignment:** **Policy Agent (Head Agent)** or individual sub-agents like the Credit Analyst.

---

## 2. Strategic Architecture: How They Connect

In the Metaopenenv multi-agent architecture, the application routes the observation data to different "Head Agents" (Risk Agent, Fraud Agent, Policy Agent). 

By connecting these two models, we can create an **Ensemble Model Pipeline**:
1. **The Qwen2.5 Model** evaluates the general financial health of the applicant.
2. **The Mistral-7B Model** evaluates the specific business/loan viability risk (especially if `Loan Type == "Business"`).

> [!TIP]
> Since these are 7B parameter models, it is possible to run them locally on consumer hardware (GPUs with 8GB-12GB VRAM if quantized, or 24GB VRAM like an RTX 3090/4090 for full 16-bit inference). Alternatively, they can be uploaded to HuggingFace and served via standard inference endpoints.

---

## 3. Connecting the Models to the Framework

Currently, agents in Metaopenenv rely on OpenRouter (e.g., `openrouter/auto`, Nvidia Nemotron, GLM-4.5) via the standard OpenAI client. To connect your locally fine-tuned models, you have two primary paths: **Local API Serving** or **Direct Python Integration**.

### Option A: Local API Serving via vLLM (Recommended)

The most robust way to integrate these models without rewriting the agent logic is to serve them locally using `vLLM` or `Ollama`, which expose an OpenAI-compatible API.

1. **Start the Mistral Server (Port 8000)**
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model d:\projects\Metaopenenv\mistral-risk-merged \
       --port 8000
   ```

2. **Start the Qwen Server (Port 8001)**
   ```bash
   python -m vllm.entrypoints.openai.api_server \
       --model d:\projects\Metaopenenv\qwen_model \
       --port 8001
   ```

3. **Update Agent Configurations (`app.py` / `agents/council/*.py`)**
   You can override the base URL of the OpenAI client passed to the specific agents.

   ```python
   from openai import OpenAI

   # Client for Mistral (Risk Agent)
   mistral_client = OpenAI(
       base_url="http://localhost:8000/v1",
       api_key="local-mistral"
   )
   
   # Client for Qwen (Policy Agent)
   qwen_client = OpenAI(
       base_url="http://localhost:8001/v1",
       api_key="local-qwen"
   )

   # Inject into Agents
   risk_agent = RiskAgent(client=mistral_client, model_name="mistral-risk-merged")
   policy_agent = PolicyAgent(client=qwen_client, model_name="qwen_model")
   ```

### Option B: Cloud Deployment via OpenRouter / HuggingFace

If you push your merged checkpoints to HuggingFace, you can utilize cloud GPU providers to serve them, or request them to be indexed by OpenRouter.

1. **Push to Hub:** (Already configured in your training scripts via `push_to_hub=True`).
2. **Use HuggingFace Inference Endpoints:** Replace the OpenRouter base URL with your dedicated HF Endpoint URL.
3. **Change the Gradio UI:** Update the "Deep Settings" Gradio UI dropdowns in `app.py` to include your new model IDs (e.g., `your-username/mistral-sba-risk`).

---

## 4. Modifying the Agent Prompts

Because these models were fine-tuned on specific instructional formats, their prompts inside the Agents should align with the training data format for maximum accuracy.

### For the Risk Agent (Mistral SBA Model)
Ensure `agents/council/risk_agent.py` passes data in the format the model expects:
```python
# Inside risk_agent.evaluate()
user_prompt = f"""
Evaluate the default risk for the following SBA Loan application:
- Loan Amount: {obs.loan_amount}
- Term: {obs.loan_term}
- Income: {obs.income}
- Mortgage Value: {obs.mortgage_value}
...
"""
```

### For the Policy Agent (Qwen Credit Model)
Ensure `agents/council/policy_agent.py` formats the sub-agent reports and consumer metrics clearly:
```python
# Inside policy_agent.evaluate()
user_prompt = f"""
Assess credit policy alignment.
Applicant Metrics: Age {obs.age}, Debt {obs.existing_debt}, Income {obs.income}.
Sub-agent reports:
{reports_str}
"""
```

## 5. Next Steps and Testing

1. **Verify Local Inference:** Run the `scripts/test_offline.py` or create a small scratch script to ping both local models to ensure they respond with the expected JSON/String format.
2. **Update UI:** Add "Local Mistral Risk" and "Local Qwen Policy" to the `app.py` Gradio Dropdowns under the "Deep Settings" tab.
3. **Run Backtests:** Use the `env/data_generator.py` to generate 100 sample applications and compare the output of the fine-tuned models vs. the generic OpenRouter models to measure accuracy improvements.

> [!IMPORTANT]
> If you run both models simultaneously on a single local machine, ensure you have sufficient VRAM. Two 7B models in 4-bit quantization will require approximately 12-14GB of VRAM combined. If you run out of memory, test them one at a time.
