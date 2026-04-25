"""
Gradio + FastAPI Application for the Loan Approval OpenEnv.

Combines:
  - FastAPI REST endpoints on port 8000
  - Gradio interactive UI on port 7860

Features:
  - Applicant profile display (financial + behavioral)
  - Sub-agent report panels with accordion drill-down
  - Council vote visualization with dissent badges
  - Big Brain final decision with override detection
  - Dissent log viewer
  - Portfolio health dashboard
  - Big Brain training status
  - Demo mode banner when API keys not available
"""

import json
import os
import sys
import threading
from typing import Optional

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
from brain.big_brain import BigBrainAgent
from brain.state_builder import PortfolioState, build_state_vector
from env.action import LoanAction
from env.loan_env import LoanApprovalEnv
from env.observation import LoanObservation

# ═══════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════

env = LoanApprovalEnv(task_level="hard", seed=42)
brain = BigBrainAgent(state_dim=28, epsilon=0.1)
brain.load("brain/big_brain_weights.pt")

current_council: Optional[VoteResult] = None
last_override_info: str = "No overrides yet"

DEMO_MODE = not (
    bool(os.getenv("NEMOTRON_API_KEY")) and 
    bool(os.getenv("GLM_API_KEY")) and 
    bool(os.getenv("MINIMAX_API_KEY"))
)

# ═══════════════════════════════════════════════════════════════
# FASTAPI BACKEND
# ═══════════════════════════════════════════════════════════════

api = FastAPI(title="Loan Approval OpenEnv API", version="1.0")


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


def run_api():
    """Run FastAPI server in a daemon thread."""
    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="warning")


# ═══════════════════════════════════════════════════════════════
# GRADIO HELPERS
# ═══════════════════════════════════════════════════════════════


def format_financial_table(obs: LoanObservation) -> list:
    """Format financial fields as table rows."""
    return [
        ["Income", f"₹{obs.income:,.0f}"],
        ["Credit Score", str(obs.credit_score)],
        ["Employment Years", str(obs.employment_years)],
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
        elif r.agent_name in ["PatternChecker", "ConsistencyChecker"]:
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


def process_next_applicant(selected_action: str, risk_model: str, fraud_model: str, policy_model: str):
    """Process one complete episode through the full pipeline."""
    global current_council, last_override_info

    # Apply dynamic model configs
    env.risk_agent.model_name = risk_model
    env.fraud_agent.model_name = fraud_model
    env.policy_agent.model_name = policy_model

    obs = env.reset()

    # Check mortgage mode
    is_mortgage_mode = obs.mortgage_value >= 1.1 * obs.loan_amount

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
            f"🏠 MORTGAGE_MODE — AUTO-APPROVED\nCollateral ₹{obs.mortgage_value:,.0f} ≥ 1.1× Loan ₹{obs.loan_amount:,.0f}",
            reward,
            "Mortgage mode: collateral sufficient, auto-approved.",
            get_dissent_log(),
            env.portfolio.total_approved,
            env.portfolio.total_rejected,
            env.portfolio.total_reviewed,
            env.portfolio.avg_reward_last_10,
            brain.steps,
            brain.epsilon,
            len(brain.replay_buffer),
            last_override_info,
        )

    # Full pipeline
    council = env.get_council_recommendation()
    current_council = council

    # Build state and get Big Brain decision
    state_vec = build_state_vector(obs, council, env.portfolio)
    is_auto_action = not selected_action or selected_action == "Auto (Big Brain)"

    if not is_auto_action:
        action = LoanAction(selected_action)
    else:
        action_str = brain.decide(state_vec)
        action = LoanAction(action_str)

    # Step
    next_obs, reward, done, info = env.step(action, council_result=council)

    # Train the RL agent on every action
    if env.current_obs is not None:
        next_council = env.get_council_recommendation()
        next_state = build_state_vector(next_obs, next_council, env.portfolio)
        brain.remember(state_vec, action.value, reward, next_state)
        brain.learn()
        brain.reward_history.append(reward)

    # Override detection
    council_rec = council.council_recommendation.value
    if action.value != council_rec:
        last_override_info = (
            f"Episode {env.episode - 1}: Brain voted {action.value}, "
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
    risk_text = f"{vote_bd.get('RiskAgent', 'N/A')} ({council.confidence_average:.0%})"
    fraud_text = f"{vote_bd.get('FraudAgent', 'N/A')} ({council.confidence_average:.0%})"
    policy_text = f"{vote_bd.get('PolicyAgent', 'N/A')} ({council.confidence_average:.0%})"

    if council.dissent_log:
        dissent_badge = f" 🔴 DISSENT: {council.dissent_log.dissenting_agent}"
    else:
        dissent_badge = ""

    brain_text = (
        f"⚖️ BIG BRAIN FINAL DECISION: {action.value}\n"
        f"{override_text}\n"
        f"Council: {council_rec} | Brain: {action.value}{dissent_badge}"
    )

    reason_text = info.get("reward_output", None)
    reason_str = reason_text.reason if reason_text else "N/A"

    # Portfolio fraud catch rate
    total_decisions = (
        env.portfolio.total_approved
        + env.portfolio.total_rejected
        + env.portfolio.total_reviewed
    )

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
        brain.steps,
        brain.epsilon,
        len(brain.replay_buffer),
        last_override_info,
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

def create_gradio_app() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(
        title="🏦 AI Loan Approval System — Big Brain v0",
        theme=gr.themes.Glass(),
    ) as demo:
        # ── HEADER ──────────────────────────────────────────
        if DEMO_MODE:
            gr.Markdown(
                "## ⚠️ DEMO MODE — No OpenRouter API keys detected. "
                "Head agents using rule-based fallback."
            )

        gr.Markdown(
            "# 🏦 AI Loan Approval System — Big Brain DQN v0\n"
            "**Three-level agent hierarchy: "
            "Sub-Agents → Head Council → Big Brain DQN**"
        )

        # ── PANEL 1: Applicant Profile ──────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Financial Data")
                financial_df = gr.Dataframe(
                    headers=["Field", "Value"],
                    label="Financial Profile",
                    interactive=False,
                )
            with gr.Column(scale=1):
                gr.Markdown("### 🧠 Behavioral Signals")
                behavioral_df = gr.Dataframe(
                    headers=["Field", "Value"],
                    label="Behavioral Profile",
                    interactive=False,
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
                )
            with gr.Accordion("🕵️ Fraud Sub-Agents", open=False):
                fraud_sub_df = gr.Dataframe(
                    headers=["Agent", "Rec", "Conf", "Flags"],
                    label="Fraud",
                    interactive=False,
                )
            with gr.Accordion("📜 Policy Sub-Agents", open=False):
                policy_sub_df = gr.Dataframe(
                    headers=["Agent", "Rec", "Conf", "Flags"],
                    label="Policy",
                    interactive=False,
                )

        # ── PANEL 3: Council Vote ───────────────────────────
        gr.Markdown("---")
        gr.Markdown("### 🗳️ Council Vote")
        with gr.Row():
            risk_vote_box = gr.Textbox(
                label="🔴 Risk Agent", interactive=False
            )
            fraud_vote_box = gr.Textbox(
                label="🟡 Fraud Agent", interactive=False
            )
            policy_vote_box = gr.Textbox(
                label="🔵 Policy Agent", interactive=False
            )

        # ── PANEL 4: Big Brain Decision ─────────────────────
        gr.Markdown("---")
        gr.Markdown("### 🧠 Big Brain Decision")
        brain_decision_box = gr.Textbox(
            label="⚖️ BIG BRAIN FINAL DECISION",
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

        # ── PANEL 7: Big Brain Status ───────────────────────
        gr.Markdown("---")
        gr.Markdown("### 🤖 Big Brain Status")
        with gr.Row():
            steps_box = gr.Number(
                label="🎓 Training Steps", interactive=False
            )
            epsilon_box = gr.Number(
                label="🎲 Epsilon", interactive=False
            )
            buffer_box = gr.Number(
                label="💾 Replay Buffer", interactive=False
            )
            override_box = gr.Textbox(
                label="⚡ Last Override", interactive=False
            )

        # ── PANEL 8: Advanced Model Configuration ───────────
        gr.Markdown("---")
        with gr.Accordion("⚙️ Advanced Model Configuration", open=False):
            gr.Markdown("Configure the OpenRouter model acting as each specific Loan Agent.")
            with gr.Row():
                risk_model_dropdown = gr.Dropdown(
                    choices=["nvidia/nemotron-4-340b-instruct:free", "z-ai/glm-4.5-air:free", "minimax/minimax-m2.5:free", "google/gemini-2.5-flash:free", "meta-llama/llama-3.3-70b-instruct:free"],
                    value="nvidia/nemotron-4-340b-instruct:free",
                    label="Risk Agent Model",
                    allow_custom_value=True
                )
                fraud_model_dropdown = gr.Dropdown(
                    choices=["nvidia/nemotron-4-340b-instruct:free", "z-ai/glm-4.5-air:free", "minimax/minimax-m2.5:free", "google/gemini-2.5-flash:free", "meta-llama/llama-3.3-70b-instruct:free"],
                    value="z-ai/glm-4.5-air:free",
                    label="Fraud Agent Model",
                    allow_custom_value=True
                )
                policy_model_dropdown = gr.Dropdown(
                    choices=["nvidia/nemotron-4-340b-instruct:free", "z-ai/glm-4.5-air:free", "minimax/minimax-m2.5:free", "google/gemini-2.5-flash:free", "meta-llama/llama-3.3-70b-instruct:free"],
                    value="minimax/minimax-m2.5:free",
                    label="Policy Agent Model",
                    allow_custom_value=True
                )

        # ── CONTROLS ────────────────────────────────────────
        gr.Markdown("---")
        with gr.Row():
            action_radio = gr.Radio(
                choices=[
                    "Auto (Big Brain)",
                    "APPROVE",
                    "REJECT",
                    "REQUEST_MORE_INFO",
                ],
                value="Auto (Big Brain)",
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
        next_btn.click(
            fn=process_next_applicant,
            inputs=[action_radio, risk_model_dropdown, fraud_model_dropdown, policy_model_dropdown],
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


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Initialize first observation
    env.reset()

    # Start FastAPI in daemon thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    print("FastAPI running on http://0.0.0.0:8000")

    # Start Gradio
    demo = create_gradio_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )