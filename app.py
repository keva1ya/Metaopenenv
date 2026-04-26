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
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

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

DEMO_MODE = not (
    bool(_env_secret("NEMOTRON_API_KEY") or _env_secret("OPENROUTER_API_KEY")) and
    bool(_env_secret("GLM_API_KEY") or _env_secret("OPENROUTER_API_KEY")) and
    bool(_env_secret("MINIMAX_API_KEY") or _env_secret("OPENROUTER_API_KEY"))
)

# ═══════════════════════════════════════════════════════════════
# FASTAPI BACKEND
# ═══════════════════════════════════════════════════════════════

api = FastAPI(title="Loan Approval OpenEnv API", version="1.0")
APP_HOST = "127.0.0.1"
DEFAULT_API_PORT = int(os.getenv("APP_API_PORT", "8000"))
DEFAULT_UI_PORT = int(os.getenv("APP_UI_PORT", "7860"))


@api.post("/reset")
def api_reset():
    """Reset environment and return new observation."""
    obs = env.reset()
    return JSONResponse(content=obs.model_dump())


@api.post("/step")
def api_step(body: dict):
    """Execute one step with the given action."""
    action_str = body.get("action", "REJECT")
    
    # Validate action input
    try:
        action = LoanAction(action_str)
    except ValueError:
        valid_actions = [a.value for a in LoanAction]
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid action",
                "message": f"Expected one of {valid_actions}, got '{action_str}'",
                "valid_actions": valid_actions
            }
        )
    
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
def api_state():
    """Return current observation."""
    if env.current_obs is None:
        return JSONResponse(content={"error": "No observation. Call /reset first."})
    return JSONResponse(content=env.current_obs.model_dump())


@api.get("/health")
def api_health():
    """Health check endpoint."""
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
    selected_action: str, 
    risk_weight: int,
    fraud_weight: int,
    policy_weight: int
):
    """Process one complete episode through the full pipeline."""
    global current_council, last_override_info

    if (risk_weight + fraud_weight + policy_weight) != 100:
        raise gr.Error("The Agent Scoring Multipliers must sum exactly to 100.")

    obs = env.reset()

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

APP_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.red,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.slate,
)

APP_CSS = """
:root {
  --brand-deep: #660000;
  --brand-ink: #2a0000;
  --brand-rose: #8f4d4d;
  --paper: #0a0a0a;
  --ink: #eaeaea;
  --muted: #888888;
  --line: rgba(102, 0, 0, 0.4);
  --panel: rgba(18, 18, 18, 0.95);
  --shadow: 0 14px 36px rgba(0, 0, 0, 0.9);
}

body {
  background: #000000 !important;
  color: var(--ink);
}

.gradio-container {
  background: transparent !important;
  max-width: 1400px !important;
  padding: 28px 18px 56px !important;
}

.gradio-container .block,
.gradio-container .gr-box,
.gradio-container .gr-group,
.gradio-container .gr-form,
.gradio-container .gr-panel,
.gradio-container .gr-accordion,
.gradio-container .gradio-dataframe,
.gradio-container .gradio-textbox,
.gradio-container .gradio-number,
.gradio-container .gradio-json {
  border-radius: 0px !important;
}

.gradio-container .gr-box,
.gradio-container .gr-group,
.gradio-container .gr-form,
.gradio-container .gr-panel,
.gradio-container .gr-accordion,
.gradio-container .gradio-dataframe,
.gradio-container .gradio-textbox,
.gradio-container .gradio-number,
.gradio-container .gradio-json,
.gradio-container .gradio-dropdown,
.gradio-container .gradio-radio {
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  box-shadow: var(--shadow) !important;
}

.hero-shell {
  margin-bottom: 18px;
}

.hero-card {
  padding: 28px 30px;
  border-radius: 0px;
  background: linear-gradient(135deg, rgba(43, 31, 31, 0.96), rgba(76, 39, 40, 0.94) 62%, rgba(108, 28, 31, 0.9));
  color: #fff7f2;
  box-shadow: 0 20px 48px rgba(58, 17, 18, 0.14);
  border: 1px solid rgba(255,255,255,0.06);
}

.hero-kicker {
  letter-spacing: 0.24em;
  text-transform: uppercase;
  font-size: 12px;
  color: #d8c1bc;
  margin-bottom: 14px;
}

.hero-title {
  font-size: clamp(2rem, 4vw, 3.5rem);
  line-height: 1.02;
  font-weight: 650;
  margin: 0 0 12px;
  max-width: 720px;
}

.hero-copy {
  max-width: 760px;
  color: #eededa;
  font-size: 0.98rem;
  line-height: 1.55;
}

.hero-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 24px;
}

.hero-pill {
  padding: 12px 14px;
  border-radius: 18px;
  background: rgba(255, 248, 243, 0.05);
  border: 1px solid rgba(255, 235, 230, 0.1);
}

.hero-pill strong {
  display: block;
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #dec1bb;
  margin-bottom: 4px;
}

.hero-pill span {
  font-size: 1rem;
  color: #fff7f2;
}

.demo-banner > div {
  background: linear-gradient(90deg, rgba(102,0,0,0.16), rgba(102,0,0,0.05)) !important;
  border: 1px solid rgba(102,0,0,0.18) !important;
  border-radius: 20px !important;
  color: var(--brand-ink) !important;
}

.gradio-container h3 {
  color: var(--brand-ink) !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em;
}

.gradio-container .gr-dataframe,
.gradio-container .gradio-dataframe {
  border-radius: 0px !important;
}

.financial-profile-panel,
.behavioral-profile-panel,
.financial-profile-column,
.behavioral-profile-column,
.financial-profile-panel > div,
.behavioral-profile-panel > div,
.financial-profile-column > div,
.behavioral-profile-column > div,
.financial-profile-panel .gradio-dataframe,
.behavioral-profile-panel .gradio-dataframe,
.financial-profile-panel .gr-dataframe,
.behavioral-profile-panel .gr-dataframe,
.financial-profile-column .gr-box,
.behavioral-profile-column .gr-box,
.financial-profile-column .gr-panel,
.behavioral-profile-column .gr-panel {
    border-radius: 0px !important;
}

.financial-profile-column *,
.behavioral-profile-column * {
    border-radius: 0px !important;
}





.gradio-container hr {
  border: 0 !important;
  height: 1px !important;
  background: linear-gradient(90deg, transparent, rgba(102, 0, 0, 0.22), transparent) !important;
  margin: 22px 0 16px !important;
}

.gradio-container .gr-button-primary,
.gradio-container button.primary {
  background: linear-gradient(135deg, #660000, #8a1c1c) !important;
  border: none !important;
  color: #fff8f3 !important;
  border-radius: 999px !important;
  box-shadow: 0 16px 34px rgba(102, 0, 0, 0.28) !important;
}

.gradio-container button:not(.primary) {
  border-radius: 999px !important;
}

.gradio-container .gr-accordion summary {
  background: rgba(102, 0, 0, 0.04) !important;
  color: var(--brand-ink) !important;
}

.gradio-container .gr-dataframe table thead tr {
  background: rgba(102, 0, 0, 0.07) !important;
}

.gradio-container .gr-dataframe table th {
  color: var(--brand-ink) !important;
}

@media (max-width: 900px) {
  .hero-card {
    padding: 26px 22px;
  }
  .hero-grid {
    grid-template-columns: 1fr;
  }
}
"""


def create_gradio_app() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(
        title="🏦 AI Loan Approval System — Dynamic Weights v1",
    ) as demo:
        # ── HEADER ──────────────────────────────────────────
        if DEMO_MODE:
            gr.Markdown(
                "## ⚠️ DEMO MODE — No OpenRouter API keys detected. "
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
                  <div class="hero-pill"><strong>Palette</strong><span>Premium scheme built around #660000</span></div>
                  <div class="hero-pill"><strong>Context</strong><span>AI-based loan analysis, fraud screening, policy review</span></div>
                </div>
              </div>
            </section>
            """
        )

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
                risk_weight_slider = gr.Number(
                    minimum=0, maximum=100, value=33, label="Risk Multiplier"
                )
                fraud_weight_slider = gr.Number(
                    minimum=0, maximum=100, value=33, label="Fraud Multiplier"
                )
                policy_weight_slider = gr.Number(
                    minimum=0, maximum=100, value=34, label="Policy Multiplier"
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
                "🚀 Next Applicant", variant="primary", size="lg"
            )
            reset_btn = gr.Button("🔄 Reset Portfolio", size="sm")
            export_btn = gr.Button("📤 Export Dissent", size="sm")

        export_output = gr.Textbox(
            label="Exported Dissent JSON",
            visible=False,
        )

        # ── WIRING ──────────────────────────────────────────

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

    api_port = _pick_port(APP_HOST, DEFAULT_API_PORT)
    ui_port = _pick_port(APP_HOST, DEFAULT_UI_PORT)

    # Start FastAPI in daemon thread
    api_thread = threading.Thread(target=run_api, args=(api_port,), daemon=True)
    api_thread.start()
    print(f"FastAPI running on http://{APP_HOST}:{api_port}")
    print(f"Gradio UI starting on http://{APP_HOST}:{ui_port}")

    # Start Gradio
    demo.launch(
        server_name=APP_HOST,
        server_port=ui_port,
        share=False,
        theme=APP_THEME,
        css=APP_CSS,
    )
