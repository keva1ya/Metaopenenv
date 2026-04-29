"""
Gradio + FastAPI Application for the Loan Approval OpenEnv.

Combines:
  - FastAPI REST endpoints on port 8000
  - Gradio interactive UI on port 7860

Features:
  - Applicant profile display (financial + behavioral)
  - Sub-agent report panels with accordion drill-down
  - Council vote visualization with dissent badges
  - Weighted final decision with override detection
  - Dissent log viewer
  - Portfolio health dashboard
  - Weighted-engine trace training status
  - Demo mode banner when API keys not available
"""

import json
import os
import socket
import sys
import threading
from typing import List, Optional

# Avoid runtime pyi generation in Gradio on startup (can stall on some environments).
os.environ.setdefault("GRADIO_SKIP_RUNTIME_PYI", "1")

import gradio as gr
import numpy as np
import pandas as pd
import io
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from agents.voting import VoteResult
from brain.dynamic_weight_net import DynamicWeightDecisionEngine
from brain.grpo_trainer import GRPOConfig, GRPOTrainerScaffold
from brain.state_builder import PortfolioState
from brain.trace_buffer import TraceBuffer, TraceRecord
from brain.reward_shaper import (
    _financial_reward,
    _fraud_reward,
    _policy_reward,
)
from env.action import LoanAction
from env.loan_env import LoanApprovalEnv
from env.observation import LoanObservation


def _env_secret(name: str) -> str:
    """Load env var and normalize surrounding whitespace/quotes."""
    raw = os.getenv(name, "")
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value


def _is_port_free(host: str, port: int) -> bool:
    """Return True when the given host/port is available for binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _pick_port(host: str, preferred_port: int, search_window: int = 50) -> int:
    """Pick the preferred port when free, otherwise the next available port."""
    for port in range(preferred_port, preferred_port + search_window):
        if _is_port_free(host, port):
            return port
    raise RuntimeError(
        f"No free port found for {host} in range "
        f"{preferred_port}-{preferred_port + search_window - 1}."
    )

# ═══════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════

env = LoanApprovalEnv(task_level="hard", seed=42)
decision_engine = DynamicWeightDecisionEngine(
    feature_dim=28,
    use_neural_weights=os.getenv("DYNAMIC_WEIGHT_USE_NEURAL", "false").lower() == "true",
)

trace_buffer = TraceBuffer(
    trace_path=os.getenv("RL_TRACE_PATH", "logs/rl_trace.jsonl"),
    max_buffer_size=int(os.getenv("RL_TRACE_MAX_BUFFER", "2048")),
    flush_every=int(os.getenv("RL_TRACE_FLUSH_EVERY", "32")),
    enabled=os.getenv("RL_TRACE_ENABLED", "true").lower() == "true",
)

grpo_trainer = GRPOTrainerScaffold(
    GRPOConfig(
        enabled=os.getenv("GRPO_TRAIN_ENABLED", "false").lower() == "true",
        train_every_steps=int(os.getenv("GRPO_TRAIN_EVERY_STEPS", "16")),
        min_trace_before_train=int(os.getenv("GRPO_MIN_TRACE_BEFORE_TRAIN", "256")),
        reward_floor=float(os.getenv("REWARD_FLOOR", "0.0")),
        uncertainty_penalty_enabled=os.getenv(
            "UNCERTAINTY_PENALTY_ENABLED", "false"
        ).lower()
        == "true",
        uncertainty_penalty_weight=float(
            os.getenv("UNCERTAINTY_PENALTY_WEIGHT", "0.0")
        ),
        reward_mode=os.getenv("REWARD_MODE", "non_negative_v1"),
    )
)
trace_history: List[TraceRecord] = []

current_council: Optional[VoteResult] = None
last_override_info: str = "No overrides yet"

dataset_df = None
dataset_index = 0


def _agent_configured(prefix: str) -> bool:
    """A council agent is live if HF (per-agent or shared) is set."""
    has_hf = bool(_env_secret(f"{prefix}_API_KEY") or _env_secret("HF_TOKEN"))
    return has_hf


DEMO_MODE = not (
    _agent_configured("RISK")
    and _agent_configured("FRAUD")
    and _agent_configured("POLICY")
)

# ═══════════════════════════════════════════════════════════════
# FASTAPI BACKEND
# ═══════════════════════════════════════════════════════════════

api = FastAPI(title="Loan Approval OpenEnv API", version="1.0")

# 1. CORS Middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Rate Limiting
limiter = Limiter(key_func=get_remote_address)
api.state.limiter = limiter
api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 3. API Key Authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    expected_api_key = os.getenv("APP_API_KEY")
    if not expected_api_key:
        return None  # Auth disabled if no key is set in env
    if api_key_header == expected_api_key:
        return api_key_header
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API Key",
    )

APP_HOST = "0.0.0.0"
DEFAULT_API_PORT = int(os.getenv("APP_API_PORT", "8000"))
DEFAULT_UI_PORT = int(os.getenv("APP_UI_PORT", "7860"))

# 4. Input Validation Models
class StepRequest(BaseModel):
    action: LoanAction


@api.post("/reset")
@limiter.limit("10/minute")
def api_reset(request: Request, api_key: str = Depends(get_api_key)):
    """Reset environment and return new observation."""
    obs = env.reset()
    return JSONResponse(content=obs.model_dump())


@api.post("/step")
@limiter.limit("30/minute")
def api_step(body: StepRequest, request: Request, api_key: str = Depends(get_api_key)):
    """Execute one step with the given action."""
    action = body.action
    
    next_obs, reward, done, info = env.step(action)
    return JSONResponse(
        content={
            "next_obs": next_obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": {
                k: v
                for k, v in info.items()
                if k != "reward_output"
            },
        }
    )


@api.get("/state")
@limiter.limit("60/minute")
def api_state(request: Request, api_key: str = Depends(get_api_key)):
    """Return current observation."""
    if env.current_obs is None:
        return JSONResponse(content={"error": "No observation. Call /reset first."})
    return JSONResponse(content=env.current_obs.model_dump())


@api.get("/health")
@limiter.limit("60/minute")
def api_health(request: Request):
    """Health check endpoint (no auth required)."""
    return JSONResponse(
        content={
            "status": "ok",
            "episode": env.episode,
            "demo_mode": DEMO_MODE,
        }
    )


def run_api(port: int):
    """Run FastAPI server in a daemon thread."""
    uvicorn.run(api, host=APP_HOST, port=port, log_level="warning")


# ═══════════════════════════════════════════════════════════════
# GRADIO HELPERS
# ═══════════════════════════════════════════════════════════════


def format_financial_table(obs: LoanObservation) -> list:
    """Format financial fields as table rows."""
    return [
        ["Income", f"₹{obs.income:,.0f}"],
        ["Credit Score", str(obs.credit_score)],
        ["Employment Years", str(obs.employment_years)],
        ["Loan Type", obs.loan_type],
        ["Loan Amount", f"₹{obs.loan_amount:,.0f}"],
        ["Mortgage Value", f"₹{obs.mortgage_value:,.0f}"],
        ["Loan Term", f"{obs.loan_term} years"],
        ["Existing Debt", f"₹{obs.existing_debt:,.0f}"],
        ["Age", str(obs.age)],
        ["Monthly Expenses", f"₹{obs.monthly_expenses:,.0f}"],
        ["Dependents", str(obs.num_dependents)],
        ["Previous Defaults", str(obs.previous_defaults)],
        ["GST Registered", "✅ Yes" if obs.gst_registered else "❌ No"],
        ["ITR Filed Years", str(obs.itr_filed_years)],
        ["Customer Score", f"{obs.customer_score:.2f}"],
    ]


def format_behavioral_table(obs: LoanObservation) -> list:
    """Format behavioral fields as table rows."""
    return [
        ["Spending Volatility", f"{obs.spending_volatility:.3f}"],
        ["Loan Purpose Clarity", f"{obs.loan_purpose_clarity:.3f}"],
        ["Application Consistency", f"{obs.application_consistency:.3f}"],
        ["Bank KYC Status", obs.bank_kyc_status],
    ]


def format_sub_reports(council: VoteResult) -> tuple:
    """Format sub-agent reports into three tables (risk, fraud, policy)."""
    risk_rows = []
    fraud_rows = []
    policy_rows = []

    for r in council.all_sub_reports:
        row = [
            r.agent_name,
            r.recommendation,
            f"{r.confidence:.2f}",
            ", ".join(r.key_flags[:3]),
        ]
        if r.agent_name in ["CreditAnalyst", "IncomeVerifier", "DebtAnalyzer"]:
            risk_rows.append(row)
        elif r.agent_name in ["PatternChecker", "ConsistencyChecker", "KYCVerifier"]:
            fraud_rows.append(row)
        else:
            policy_rows.append(row)

    return risk_rows, fraud_rows, policy_rows


def get_dissent_log() -> list:
    """Load dissent log entries."""
    path = "logs/dissent_log.json"
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            logs = json.load(f)
        return logs[-10:] if logs else []
    except (json.JSONDecodeError, IOError):
        return []


# ═══════════════════════════════════════════════════════════════
# MAIN GRADIO ACTION
# ═══════════════════════════════════════════════════════════════


def process_next_applicant(
    # 19 fields
    income, credit_score, employment_years, loan_amount, loan_type, mortgage_value,
    loan_term, existing_debt, age, monthly_expenses, num_dependents, previous_defaults,
    gst_registered, itr_filed_years, customer_score, spending_volatility,
    loan_purpose_clarity, application_consistency, bank_kyc_status,
    # action & weights
    selected_action: str, 
    risk_weight: int,
    fraud_weight: int,
    policy_weight: int,
):
    global current_council, last_override_info

    if (risk_weight + fraud_weight + policy_weight) != 100:
        raise gr.Error("Scoring multipliers must sum to exactly 100.")

    try:
        obs = LoanObservation(
            income=float(income),
            credit_score=int(credit_score),
            employment_years=int(employment_years),
            loan_amount=float(loan_amount),
            loan_type=str(loan_type),
            mortgage_value=float(mortgage_value),
            loan_term=int(loan_term),
            existing_debt=float(existing_debt),
            age=int(age),
            monthly_expenses=float(monthly_expenses),
            num_dependents=int(num_dependents),
            previous_defaults=int(previous_defaults),
            gst_registered=bool(gst_registered),
            itr_filed_years=int(itr_filed_years),
            customer_score=float(customer_score),
            spending_volatility=float(spending_volatility),
            loan_purpose_clarity=float(loan_purpose_clarity),
            application_consistency=float(application_consistency),
            bank_kyc_status=str(bank_kyc_status)
        )
    except Exception as e:
        raise gr.Error(f"Invalid input data: {e}")

    env.set_observation(obs)
    env.episode += 1

    # Check mortgage mode
    is_mortgage_mode = obs.mortgage_value >= 1.3 * obs.loan_amount

    if is_mortgage_mode:
        action = LoanAction.APPROVE
        next_obs, reward, done, info = env.step(action)

        financial_table = format_financial_table(obs)
        behavioral_table = format_behavioral_table(obs)

        return (
            financial_table,
            behavioral_table,
            [], [], [],  # sub-agent tables
            "🏠 MORTGAGE_MODE", "🏠 MORTGAGE_MODE", "🏠 MORTGAGE_MODE",
            f"🏠 MORTGAGE_MODE — AUTO-APPROVED\nCollateral ₹{obs.mortgage_value:,.0f} ≥ 1.3× Loan ₹{obs.loan_amount:,.0f}",
            reward,
            "Mortgage mode: collateral sufficient, auto-approved.",
            get_dissent_log(),
            env.portfolio.total_approved,
            env.portfolio.total_rejected,
            env.portfolio.total_reviewed,
            env.portfolio.avg_reward_last_10,
            len(trace_history),
            float(grpo_trainer.last_status == "trained_placeholder"),
            trace_buffer.pending_count(),
            last_override_info,
        )

    # Full pipeline
    council = env.get_council_recommendation()
    current_council = council

    is_auto_action = (
        not selected_action
        or selected_action == "Auto (Dynamic Weights)"
    )

    if not is_auto_action:
        action = LoanAction(selected_action)
        decision_result = None
    else:
        decision_result = decision_engine.decide(
            obs=obs,
            council=council,
            portfolio=env.portfolio,
            manual_multipliers={
                "RiskAgent": float(risk_weight),
                "FraudAgent": float(fraud_weight),
                "PolicyAgent": float(policy_weight),
            },
        )
        action = decision_result.action

    # Step
    next_obs, reward, done, info = env.step(action, council_result=council)

    if decision_result is not None:
        reward_financial = _financial_reward(obs, action.value)
        reward_fraud = _fraud_reward(obs, action.value)
        reward_policy = _policy_reward(obs, action.value)
        reward_alignment = (
            1.0 if action.value == council.council_recommendation.value else 0.0
        )
        reward_total = max(
            float(grpo_trainer.config.reward_floor),
            float(0.4 * reward_financial + 0.4 * reward_fraud + 0.2 * reward_policy),
        )

        trace = TraceRecord(
            episode=int(env.episode - 1),
            final_action=action.value,
            ground_truth_label=None,
            reward_total=reward_total,
            reward_confidence=float(council.confidence_average),
            reward_alignment=reward_alignment,
            uncertainty_cost_applied=0.0,
            uncertainty_score=float(decision_result.uncertainty_score),
            penalty_mode="disabled"
            if not grpo_trainer.config.uncertainty_penalty_enabled
            else "reserved",
            penalty_config_version=grpo_trainer.config.reward_mode,
            head_votes=dict(council.vote_breakdown),
            head_confidences=dict(council.agent_confidence),
            head_reasoning=dict(council.agent_reasoning),
            dynamic_weights=dict(decision_result.normalized_weights),
            weighted_action_scores=dict(decision_result.weighted_action_scores),
            diagnostics={
                "risk_multiplier": float(risk_weight),
                "fraud_multiplier": float(fraud_weight),
                "policy_multiplier": float(policy_weight),
            },
        )
        trace_history.append(trace)
        trace_buffer.push(trace)
        grpo_trainer.maybe_train(trace_history)

    # Override detection
    council_rec = council.council_recommendation.value
    if action.value != council_rec:
        last_override_info = (
            f"Episode {env.episode - 1}: Engine voted {action.value}, "
            f"council recommended {council_rec}"
        )
        override_text = f"⚠️ Overrode council vote ({council_rec} → {action.value})"
    else:
        override_text = f"✅ Agreed with council ({council_rec})"

    # Format outputs
    financial_table = format_financial_table(obs)
    behavioral_table = format_behavioral_table(obs)
    risk_rows, fraud_rows, policy_rows = format_sub_reports(council)

    vote_bd = council.vote_breakdown
    agent_rs = council.agent_reasoning
    agent_cf = council.agent_confidence
    
    if decision_result is not None:
        norm_w = decision_result.normalized_weights
        risk_score = norm_w.get("RiskAgent", 0.0)
        fraud_score = norm_w.get("FraudAgent", 0.0)
        policy_score = norm_w.get("PolicyAgent", 0.0)
    else:
        risk_score = agent_cf.get("RiskAgent", 0.0)
        fraud_score = agent_cf.get("FraudAgent", 0.0)
        policy_score = agent_cf.get("PolicyAgent", 0.0)
    
    risk_text = f"VOTE: {vote_bd.get('RiskAgent', 'N/A')} (Weight: {risk_score:.2f})\nREASON: {agent_rs.get('RiskAgent', 'N/A')}"
    fraud_text = f"VOTE: {vote_bd.get('FraudAgent', 'N/A')} (Weight: {fraud_score:.2f})\nREASON: {agent_rs.get('FraudAgent', 'N/A')}"
    policy_text = f"VOTE: {vote_bd.get('PolicyAgent', 'N/A')} (Weight: {policy_score:.2f})\nREASON: {agent_rs.get('PolicyAgent', 'N/A')}"

    if council.dissent_log:
        dissent_badge = f" 🔴 DISSENT: {council.dissent_log.dissenting_agent}"
    else:
        dissent_badge = ""

    brain_text = (
        f"⚖️ WEIGHTED FINAL DECISION: {action.value}\n"
        f"{override_text}\n"
        f"Council: {council_rec} | Engine: {action.value}{dissent_badge}"
    )

    reason_text = info.get("reward_output", None)
    reason_str = reason_text.reason if reason_text else "N/A"

    return (
        financial_table,
        behavioral_table,
        risk_rows,
        fraud_rows,
        policy_rows,
        risk_text,
        fraud_text,
        policy_text,
        brain_text,
        reward,
        reason_str,
        get_dissent_log(),
        env.portfolio.total_approved,
        env.portfolio.total_rejected,
        env.portfolio.total_reviewed,
        env.portfolio.avg_reward_last_10,
        len(trace_history),
        1.0 if grpo_trainer.config.enabled else 0.0,
        trace_buffer.pending_count(),
        last_override_info,
    )


def show_processing_skeleton():
    """Fill the full dashboard with a polished skeleton state during processing."""
    financial_skeleton = [
        ["Applicant Profile", "Generating secured borrower snapshot"],
        ["Income", "████████████ calculating"],
        ["Credit Score", "██████ validating bureau signal"],
        ["Loan Amount", "████████ underwriting request"],
        ["Collateral", "████████ checking mortgage cover"],
        ["Existing Debt", "██████ measuring obligations"],
        ["Age / Term", "████ policy window scan"],
    ]
    behavioral_skeleton = [
        ["Spending Volatility", "████████ pattern loading"],
        ["Purpose Clarity", "██████ intent extraction"],
        ["Application Consistency", "████████ cross-checking"],
        ["Bank KYC Status", "████ identity verification"],
    ]
    risk_skeleton = [
        ["CreditAnalyst", "SCANNING", "--", "Bureau, score, repayment capacity"],
        ["IncomeVerifier", "SCANNING", "--", "Income stability and affordability"],
        ["DebtAnalyzer", "SCANNING", "--", "Debt load and repayment pressure"],
    ]
    fraud_skeleton = [
        ["PatternChecker", "SCANNING", "--", "Velocity and anomaly profile"],
        ["ConsistencyChecker", "SCANNING", "--", "Application field consistency"],
        ["IdentityVerifier", "SCANNING", "--", "KYC and identity confidence"],
    ]
    policy_skeleton = [
        ["ComplianceOfficer", "SCANNING", "--", "Bank policy constraints"],
        ["TermValidator", "SCANNING", "--", "Age plus tenure validation"],
        ["LTVChecker", "SCANNING", "--", "Loan-to-value coverage"],
    ]
    council_text = (
        "UNDERWRITING IN PROGRESS\n"
        "Signal packet received. Agent reasoning is being synthesized.\n"
        "Confidence, flags, and dissent will appear here."
    )
    brain_text = (
        "WEIGHTED ENGINE ACTIVE\n"
        "Combining risk, fraud, policy, and portfolio context.\n"
        "Final action pending."
    )
    dissent_skeleton = [
        {
            "stage": "processing",
            "message": "Council votes are being compared for dissent.",
        }
    ]

    return (
        financial_skeleton,
        behavioral_skeleton,
        risk_skeleton,
        fraud_skeleton,
        policy_skeleton,
        council_text,
        council_text,
        council_text,
        brain_text,
        0.0,
        "Reward explanation pending while the full loan decision pipeline runs.",
        dissent_skeleton,
        env.portfolio.total_approved,
        env.portfolio.total_rejected,
        env.portfolio.total_reviewed,
        env.portfolio.avg_reward_last_10,
        len(trace_history),
        1.0 if grpo_trainer.config.enabled else 0.0,
        trace_buffer.pending_count(),
        "Processing current applicant...",
    )


def reset_portfolio():
    """Reset portfolio state."""
    env.portfolio = PortfolioState()
    return (0, 0, 0, 0.0)


def export_dissent():
    """Export dissent log as formatted JSON string."""
    return json.dumps(get_dissent_log(), indent=2)


# ═══════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════



def handle_dataset_upload(file):
    global dataset_df, dataset_index
    if file is None:
        return ("No file uploaded.",) + tuple(gr.update() for _ in range(19))
    
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith('.json'):
            df = pd.read_json(file.name)
        else:
            return ("Unsupported file format.",) + tuple(gr.update() for _ in range(19))
        
        dataset_df = df
        dataset_index = 0
        return (f"Successfully loaded {len(df)} rows. Click 'Load Next Row'.",) + tuple(gr.update() for _ in range(19))
    except Exception as e:
        return (f"Error loading dataset: {e}",) + tuple(gr.update() for _ in range(19))

def load_next_from_dataset():
    global dataset_df, dataset_index
    if dataset_df is None or len(dataset_df) == 0:
        return ("No dataset loaded.",) + tuple(gr.update() for _ in range(19))
    
    if dataset_index >= len(dataset_df):
        dataset_index = 0
        
    row = dataset_df.iloc[dataset_index]
    dataset_index += 1
    
    return (
        f"Loaded row {dataset_index}/{len(dataset_df)}",
        float(row.get('income', 50000)),
        int(row.get('credit_score', 700)),
        int(row.get('employment_years', 5)),
        float(row.get('loan_amount', 100000)),
        str(row.get('loan_type', 'Personal')),
        float(row.get('mortgage_value', 0)),
        int(row.get('loan_term', 10)),
        float(row.get('existing_debt', 5000)),
        int(row.get('age', 35)),
        float(row.get('monthly_expenses', 2000)),
        int(row.get('num_dependents', 0)),
        int(row.get('previous_defaults', 0)),
        bool(row.get('gst_registered', False)),
        int(row.get('itr_filed_years', 3)),
        float(row.get('customer_score', 0.8)),
        float(row.get('spending_volatility', 0.5)),
        float(row.get('loan_purpose_clarity', 0.8)),
        float(row.get('application_consistency', 0.9)),
        str(row.get('bank_kyc_status', 'VERIFIED'))
    )

def create_gradio_app() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(
        title="🏦 AI Loan Approval System — Dynamic Weights v1",
    ) as demo:
        # ── HEADER ──────────────────────────────────────────
        if DEMO_MODE:
            gr.Markdown(
                "## ⚠️ DEMO MODE — Hugging Face secrets not detected. "
                "Head agents using rule-based fallback.",
                elem_classes=["demo-banner"],
            )

        gr.HTML(
            """
            <section class="hero-shell">
              <div class="hero-card">
                <div class="hero-kicker">AI-Based Loan Analysis System</div>
                <h1 class="hero-title">Underwriting with signal, dissent, and weighted judgment.</h1>
                <div class="hero-copy">
                  A decision cockpit for applicant review, fraud vigilance, policy synthesis,
                  and portfolio-aware recommendations. Built to feel less like a generic dashboard
                  and more like a private banking risk desk.
                </div>
                <div class="hero-grid">
                  <div class="hero-pill"><strong>Architecture</strong><span>Sub-agents -> Council -> Weighted Engine</span></div>
                  <div class="hero-pill"><strong>Security</strong><span>Secrets-based HF Authentication</span></div>
                  <div class="hero-pill"><strong>Context</strong><span>AI-based loan analysis, fraud screening, policy review</span></div>
                </div>
              </div>
            </section>
            """
        )

        

        # ── MANUAL ENTRY / DATASET INPUT ────────────────────
        gr.Markdown("---")
        gr.Markdown("### 📝 Applicant Data Entry / Dataset Upload")
        with gr.Accordion("Upload Dataset (CSV/JSON)", open=False):
            with gr.Row():
                dataset_file = gr.File(label="Upload Dataset File", file_types=[".csv", ".json"])
            with gr.Row():
                upload_status = gr.Textbox(label="Dataset Status", interactive=False)
            with gr.Row():
                load_next_btn = gr.Button("⬇️ Load Next Row from Dataset")

        with gr.Accordion("Manual Data Entry", open=True):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Financial Data")
                    inp_income = gr.Number(label="Income", value=80000)
                    inp_credit_score = gr.Number(label="Credit Score", value=750)
                    inp_employment_years = gr.Number(label="Employment Years", value=5)
                    inp_loan_amount = gr.Number(label="Loan Amount", value=200000)
                    inp_loan_type = gr.Dropdown(choices=["Personal", "Business", "Mortgage", "Auto", "Student", "Home/Land"], value="Personal", label="Loan Type")
                    inp_mortgage_value = gr.Number(label="Mortgage Value", value=0)
                    inp_loan_term = gr.Number(label="Loan Term", value=15)
                    inp_existing_debt = gr.Number(label="Existing Debt", value=10000)
                    inp_age = gr.Number(label="Age", value=35)
                    inp_monthly_expenses = gr.Number(label="Monthly Expenses", value=3000)
                with gr.Column():
                    gr.Markdown("#### Additional Info & Behavioral Data")
                    inp_num_dependents = gr.Number(label="Dependents", value=2)
                    inp_previous_defaults = gr.Number(label="Previous Defaults", value=0)
                    inp_gst_registered = gr.Checkbox(label="GST Registered", value=False)
                    inp_itr_filed_years = gr.Number(label="ITR Filed Years", value=4)
                    inp_customer_score = gr.Slider(minimum=0, maximum=1, value=0.85, label="Customer Score")
                    inp_spending_volatility = gr.Slider(minimum=0, maximum=1, value=0.4, label="Spending Volatility")
                    inp_loan_purpose_clarity = gr.Slider(minimum=0, maximum=1, value=0.9, label="Loan Purpose Clarity")
                    inp_application_consistency = gr.Slider(minimum=0, maximum=1, value=0.95, label="Application Consistency")
                    inp_bank_kyc_status = gr.Dropdown(choices=["VERIFIED", "PENDING", "FAILED"], value="VERIFIED", label="Bank KYC Status")

# ── PANEL 1: Applicant Profile ──────────────────────
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["financial-profile-column"]):
                gr.Markdown("### 📋 Financial Data")
                financial_df = gr.Dataframe(
                    headers=["Field", "Value"],
                    label="Financial Profile",
                    interactive=False,
                    column_widths=["55%", "45%"],
                    wrap=True,
                    max_height=900,
                    elem_classes=["financial-profile-panel"],
                )
            with gr.Column(scale=1, elem_classes=["behavioral-profile-column"]):
                gr.Markdown("### 🧠 Behavioral Signals")
                behavioral_df = gr.Dataframe(
                    headers=["Field", "Value"],
                    label="Behavioral Profile",
                    interactive=False,
                    column_widths=["55%", "45%"],
                    wrap=True,
                    max_height=900,
                    elem_classes=["behavioral-profile-panel"],
                )

        # ── PANEL 2: Sub-Agent Reports ──────────────────────
        gr.Markdown("---")
        gr.Markdown("### 🔍 Sub-Agent Reports")
        with gr.Row():
            with gr.Accordion("📊 Risk Sub-Agents", open=False):
                risk_sub_df = gr.Dataframe(
                    headers=["Agent", "Rec", "Conf", "Flags"],
                    label="Risk",
                    interactive=False,
                    wrap=True,
                    max_height=600,
                )
            with gr.Accordion("🕵️ Fraud Sub-Agents", open=False):
                fraud_sub_df = gr.Dataframe(
                    headers=["Agent", "Rec", "Conf", "Flags"],
                    label="Fraud",
                    interactive=False,
                    wrap=True,
                    max_height=600,
                )
            with gr.Accordion("📜 Policy Sub-Agents", open=False):
                policy_sub_df = gr.Dataframe(
                    headers=["Agent", "Rec", "Conf", "Flags"],
                    label="Policy",
                    interactive=False,
                    wrap=True,
                    max_height=600,
                )

        # ── PANEL 3: Council Vote ───────────────────────────
        gr.Markdown("---")
        gr.Markdown("### 🗳️ Council Vote")
        with gr.Row():
            risk_vote_box = gr.Textbox(
                label="🔴 Risk Agent", interactive=False, lines=6, max_lines=12
            )
            fraud_vote_box = gr.Textbox(
                label="🟡 Fraud Agent", interactive=False, lines=6, max_lines=12
            )
            policy_vote_box = gr.Textbox(
                label="🔵 Policy Agent", interactive=False, lines=6, max_lines=12
            )

        # ── PANEL 4: Weighted Engine Decision ───────────────
        gr.Markdown("---")
        gr.Markdown("### 🧠 Weighted Engine Decision")
        brain_decision_box = gr.Textbox(
            label="⚖️ WEIGHTED FINAL DECISION",
            interactive=False,
            lines=3,
        )
        with gr.Row():
            reward_box = gr.Number(
                label="💰 Reward Score", interactive=False
            )
            reason_box = gr.Textbox(
                label="📝 Reason", interactive=False
            )

        # ── PANEL 5: Dissent Log ────────────────────────────
        gr.Markdown("---")
        gr.Markdown("### 📜 Dissent Log (Last 10)")
        dissent_json = gr.JSON(label="Dissent History")

        # ── PANEL 6: Portfolio Health ───────────────────────
        gr.Markdown("---")
        gr.Markdown("### 📊 Portfolio Health")
        with gr.Row():
            approved_box = gr.Number(
                label="✅ Approved", interactive=False
            )
            rejected_box = gr.Number(
                label="❌ Rejected", interactive=False
            )
            reviewed_box = gr.Number(
                label="🔍 Reviewed", interactive=False
            )
            avg_reward_box = gr.Number(
                label="📈 Avg Reward (Last 10)",
                interactive=False,
            )

        # ── PANEL 7: Runtime Status ─────────────────────────
        gr.Markdown("---")
        gr.Markdown("### 🤖 Runtime Status")
        with gr.Row():
            steps_box = gr.Number(
                label="🧾 Trace Count", interactive=False
            )
            epsilon_box = gr.Number(
                label="🧪 GRPO Enabled (0/1)", interactive=False
            )
            buffer_box = gr.Number(
                label="💾 Trace Buffer Pending", interactive=False
            )
            override_box = gr.Textbox(
                label="⚡ Last Override", interactive=False
            )

        # ── PANEL 8: Scoring Multipliers ────────────────────
        gr.Markdown("---")
        with gr.Accordion("⚖️ Agent Scoring Multipliers (Must sum to 100)", open=True):
            gr.Markdown("Adjust the scoring weights for each head agent. These multipliers will scale the individual LLM confidence outputs.")
            with gr.Row():
                risk_weight_slider = gr.Slider(
                    minimum=0, maximum=100, step=1, value=33, label="Risk Multiplier"
                )
                fraud_weight_slider = gr.Slider(
                    minimum=0, maximum=100, step=1, value=33, label="Fraud Multiplier"
                )
                policy_weight_slider = gr.Slider(
                    minimum=0, maximum=100, step=1, value=34, label="Policy Multiplier"
                )


        # ── CONTROLS ────────────────────────────────────────
        gr.Markdown("---")
        with gr.Row():
            action_radio = gr.Radio(
                choices=[
                    "Auto (Dynamic Weights)",
                    "APPROVE",
                    "REJECT",
                    "REQUEST_MORE_INFO",
                ],
                value="Auto (Dynamic Weights)",
                label="Action Selection",
            )
        with gr.Row():
            next_btn = gr.Button(
                "🚀 Analyze", variant="primary", size="lg"
            )
            reset_btn = gr.Button("🔄 Reset Portfolio", size="sm")
            export_btn = gr.Button("📤 Export Dissent", size="sm")

        export_output = gr.Textbox(
            label="Exported Dissent JSON",
            visible=False,
        )

        # ── WIRING ──────────────────────────────────────────

        
        input_fields = [
            inp_income, inp_credit_score, inp_employment_years, inp_loan_amount, inp_loan_type,
            inp_mortgage_value, inp_loan_term, inp_existing_debt, inp_age, inp_monthly_expenses,
            inp_num_dependents, inp_previous_defaults, inp_gst_registered, inp_itr_filed_years,
            inp_customer_score, inp_spending_volatility, inp_loan_purpose_clarity,
            inp_application_consistency, inp_bank_kyc_status
        ]

        dataset_file.upload(
            fn=handle_dataset_upload,
            inputs=[dataset_file],
            outputs=[upload_status] + input_fields
        )

        load_next_btn.click(
            fn=load_next_from_dataset,
            inputs=[],
            outputs=[upload_status] + input_fields
        )

        next_flow = next_btn.click(

            fn=show_processing_skeleton,
            inputs=[],
            outputs=[
                financial_df,
                behavioral_df,
                risk_sub_df,
                fraud_sub_df,
                policy_sub_df,
                risk_vote_box,
                fraud_vote_box,
                policy_vote_box,
                brain_decision_box,
                reward_box,
                reason_box,
                dissent_json,
                approved_box,
                rejected_box,
                reviewed_box,
                avg_reward_box,
                steps_box,
                epsilon_box,
                buffer_box,
                override_box,
            ],
            queue=False,
        )

        next_flow.then(
            fn=process_next_applicant,
            inputs=[
                inp_income, inp_credit_score, inp_employment_years, inp_loan_amount, inp_loan_type,
                inp_mortgage_value, inp_loan_term, inp_existing_debt, inp_age, inp_monthly_expenses,
                inp_num_dependents, inp_previous_defaults, inp_gst_registered, inp_itr_filed_years,
                inp_customer_score, inp_spending_volatility, inp_loan_purpose_clarity,
                inp_application_consistency, inp_bank_kyc_status,
                action_radio, 
                risk_weight_slider,
                fraud_weight_slider,
                policy_weight_slider
            ],
            outputs=[
                financial_df,
                behavioral_df,
                risk_sub_df,
                fraud_sub_df,
                policy_sub_df,
                risk_vote_box,
                fraud_vote_box,
                policy_vote_box,
                brain_decision_box,
                reward_box,
                reason_box,
                dissent_json,
                approved_box,
                rejected_box,
                reviewed_box,
                avg_reward_box,
                steps_box,
                epsilon_box,
                buffer_box,
                override_box,
            ],
        )

        reset_btn.click(
            fn=reset_portfolio,
            inputs=[],
            outputs=[
                approved_box,
                rejected_box,
                reviewed_box,
                avg_reward_box,
            ],
        )

        export_btn.click(
            fn=export_dissent,
            inputs=[],
            outputs=[export_output],
        )

    return demo

demo = create_gradio_app()

# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Initialize first observation
    env.reset()

    if os.getenv("SPACE_ID"):
        # On Hugging Face Spaces, mount Gradio onto FastAPI and run on the provided PORT
        port = int(os.getenv("PORT", 7860))
        print(f"Running on Hugging Face Spaces on port {port}")
        
        # We mount Gradio onto the FastAPI app 'api'
        app = gr.mount_gradio_app(api, demo, path="/")
        
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # Local execution with separate ports
        api_port = _pick_port(APP_HOST, DEFAULT_API_PORT)
        ui_port = _pick_port(APP_HOST, DEFAULT_UI_PORT)

        # Start FastAPI in daemon thread
        api_thread = threading.Thread(target=run_api, args=(api_port,), daemon=True)
        api_thread.start()
        print(f"FastAPI running on http://{APP_HOST}:{api_port}")
        print(f"Gradio UI starting on http://{APP_HOST}:{ui_port}")

        # Start Gradio
        ui_username = os.getenv("APP_UI_USERNAME")
        ui_password = os.getenv("APP_UI_PASSWORD")
        auth_config = (ui_username, ui_password) if ui_username and ui_password else None

        demo.launch(
            server_name=APP_HOST,
            server_port=ui_port,
            share=False,
            auth=auth_config,
        )
