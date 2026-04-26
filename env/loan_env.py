"""
Main Loan Approval Environment.

Orchestrates the three-level agent hierarchy:
  Level 1: 9 sub-agents (rule-based)
  Level 2: 3 head agents (GPT-4o-mini)
  Level 3: Weighted decision engine (final decision)

Supports three task levels (easy, medium, hard)
and MORTGAGE_MODE auto-approval.
"""

import json
import os
from typing import Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv

from agents.council.fraud_agent import FraudAgent
from agents.council.policy_agent import PolicyAgent
from agents.council.risk_agent import RiskAgent
from agents.sub_agents.fraud.consistency_checker import ConsistencyCheckerAgent
from agents.sub_agents.fraud.identity_verifier import IdentityVerifierAgent
from agents.sub_agents.fraud.pattern_checker import PatternCheckerAgent
from agents.sub_agents.policy.compliance_officer import ComplianceOfficerAgent
from agents.sub_agents.policy.ltv_checker import LTVCheckerAgent
from agents.sub_agents.policy.term_validator import TermValidatorAgent
from agents.sub_agents.risk.credit_analyst import CreditAnalystAgent
from agents.sub_agents.risk.debt_analyzer import DebtAnalyzerAgent
from agents.sub_agents.risk.income_verifier import IncomeVerifierAgent
from agents.voting import VoteResult, VotingEngine
from brain.state_builder import PortfolioState
from brain.reward_shaper import calculate_shaped_reward
from env.action import LoanAction
from env.grader import EasyGrader, HardGrader, MediumGrader
from env.observation import LoanObservation

load_dotenv()


def _env_secret(name: str) -> str:
    """Load env var and normalize surrounding whitespace/quotes."""
    raw = os.getenv(name, "")
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value


def _get_fraud_score(obs: LoanObservation) -> int:
    """Calculate raw fraud score from observation."""
    score = 0
    if obs.spending_volatility > 0.7 and obs.loan_amount > 3 * obs.income:
        score += 1
    if obs.employment_years <= 1 and obs.loan_amount > 200000:
        score += 1
    if obs.income < 40000 and obs.loan_amount > 300000:
        score += 1
    return score


class LoanApprovalEnv:
    """
    OpenEnv-compliant loan approval environment.

    Manages the full pipeline:
      1. Data generation
      2. Sub-agent analysis (rule-based)
      3. Head agent voting (GPT-4o-mini / fallback)
      4. Grading and reward calculation
    """

    def __init__(
        self,
        task_level: str = "hard",
        seed: int = 42,
    ):
        self.task_level = task_level
        self.episode = 0
        self.current_obs: Optional[LoanObservation] = None
        self.portfolio = PortfolioState()
        self.dissent_context = "No dissent history yet."

        # ── ENSEMBLE MODEL PIPELINE ──────────────────────────
        # Each council head agent is wired to a fine-tuned model:
        #   RiskAgent   → Qwen 2.5 (fine-tuned on Google Colab)
        #   PolicyAgent → Mistral 7B (fine-tuned locally)
        #   FraudAgent  → Llama 3 (fine-tuned locally)
        #
        # All three are hosted on Hugging Face. Inference is done via
        # any OpenAI-compatible endpoint (HF Inference Endpoints,
        # HF Inference Providers router, or local vLLM/Ollama).
        #
        # Resolution order per agent:
        #   1. Agent-specific vars: <AGENT>_API_BASE/_API_KEY/_MODEL
        #   2. Shared HF token: HF_TOKEN with default HF router base URL
        #   3. OpenRouter legacy fallback: NEMOTRON/GLM/MINIMAX keys
        #   4. Demo mode (deterministic rule-based fallback)
        # ──────────────────────────────────────────────────────
        shared_openrouter_key = _env_secret("OPENROUTER_API_KEY")
        shared_hf_token = _env_secret("HF_TOKEN")

        DEFAULT_HF_BASE = "https://router.huggingface.co/v1"
        OPENROUTER_BASE = "https://openrouter.ai/api/v1"

        def _resolve_agent(prefix: str, legacy_key_var: str, openrouter_default_model: str):
            """
            Resolve (api_key, base_url, model_name) for a council agent.
            Prefers HF config; falls back to OpenRouter; returns None key for demo mode.
            """
            agent_key = _env_secret(f"{prefix}_API_KEY")
            agent_base = _env_secret(f"{prefix}_API_BASE")
            agent_model = _env_secret(f"{prefix}_MODEL")

            if agent_key and agent_model:
                return (
                    agent_key,
                    agent_base or DEFAULT_HF_BASE,
                    agent_model,
                )

            if shared_hf_token and agent_model:
                return (
                    shared_hf_token,
                    agent_base or DEFAULT_HF_BASE,
                    agent_model,
                )

            legacy_key = _env_secret(legacy_key_var) or shared_openrouter_key
            if legacy_key:
                return (
                    legacy_key,
                    OPENROUTER_BASE,
                    openrouter_default_model,
                )

            return (None, None, openrouter_default_model)

        risk_key, risk_base, risk_model = _resolve_agent(
            prefix="RISK",
            legacy_key_var="NEMOTRON_API_KEY",
            openrouter_default_model="nvidia/nemotron-nano-9b-v2:free",
        )
        fraud_key, fraud_base, fraud_model = _resolve_agent(
            prefix="FRAUD",
            legacy_key_var="GLM_API_KEY",
            openrouter_default_model="z-ai/glm-4.5-air:free",
        )
        policy_key, policy_base, policy_model = _resolve_agent(
            prefix="POLICY",
            legacy_key_var="MINIMAX_API_KEY",
            openrouter_default_model="minimax/minimax-m2.5:free",
        )

        self.demo_mode = not (risk_key or fraud_key or policy_key)

        def _build_client(api_key: Optional[str], base_url: Optional[str]):
            if not api_key or not base_url:
                return None
            return OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=15.0,
                max_retries=1,
            )

        risk_client = _build_client(risk_key, risk_base)
        fraud_client = _build_client(fraud_key, fraud_base)
        policy_client = _build_client(policy_key, policy_base)

        if self.demo_mode:
            print(
                "DEMO MODE: No HuggingFace or OpenRouter keys found. "
                "Head agents using rule-based fallback."
            )

        # Level 1: Sub-agents (all rule-based, no LLM)
        self.credit_analyst = CreditAnalystAgent()
        self.income_verifier = IncomeVerifierAgent()
        self.debt_analyzer = DebtAnalyzerAgent()
        self.pattern_checker = PatternCheckerAgent()
        self.consistency_checker = ConsistencyCheckerAgent()
        self.identity_verifier = IdentityVerifierAgent()
        # Policy sub-agents share the policy client (Mistral on HF)
        self.compliance_officer = ComplianceOfficerAgent(
            client=policy_client, model_name=policy_model
        )
        self.term_validator = TermValidatorAgent(
            client=policy_client, model_name=policy_model
        )
        self.ltv_checker = LTVCheckerAgent(
            client=policy_client, model_name=policy_model
        )

        # Level 2: Head agents (ensemble of fine-tuned HF models with fallback)
        self.risk_agent = RiskAgent(risk_client, model_name=risk_model)
        self.fraud_agent = FraudAgent(fraud_client, model_name=fraud_model)
        self.policy_agent = PolicyAgent(policy_client, model_name=policy_model)

        # Voting engine
        self.voting_engine = VotingEngine()

        # Grader selection
        if task_level == "easy":
            self.grader = EasyGrader()
        elif task_level == "medium":
            self.grader = MediumGrader()
        else:
            self.grader = HardGrader()

    def reset(self) -> Optional[LoanObservation]:
        """Reset environment: prepare for new episode."""
        self.episode += 1
        self.current_obs = None
        self.dissent_context = self._load_dissent_context()
        return self.current_obs

    def set_observation(self, obs: LoanObservation) -> None:
        """Manually inject user or dataset-provided observation."""
        self.current_obs = obs

    def get_council_recommendation(self) -> VoteResult:
        """
        Run the full Level 1 + Level 2 pipeline.
        Returns VoteResult which is a RECOMMENDATION, not final.
        """
        if self.current_obs is None:
            raise RuntimeError("No current observation. Call reset() first.")

        obs = self.current_obs

        # Level 1: Run all 9 sub-agents
        credit_r = self.credit_analyst.analyze(obs)
        income_r = self.income_verifier.analyze(obs)
        debt_r = self.debt_analyzer.analyze(obs)
        pattern_r = self.pattern_checker.analyze(obs)
        consistency_r = self.consistency_checker.analyze(obs)
        identity_r = self.identity_verifier.analyze(obs)
        compliance_r = self.compliance_officer.analyze(obs)
        term_r = self.term_validator.analyze(obs)
        ltv_r = self.ltv_checker.analyze(obs)

        all_sub_reports = [
            credit_r, income_r, debt_r,
            pattern_r, consistency_r, identity_r,
            compliance_r, term_r, ltv_r,
        ]

        # Level 2: Run head agents
        risk_vote = self.risk_agent.evaluate(
            obs, [credit_r, income_r, debt_r], self.dissent_context
        )
        fraud_vote = self.fraud_agent.evaluate(
            obs, [pattern_r, consistency_r, identity_r], self.dissent_context
        )
        policy_vote = self.policy_agent.evaluate(
            obs, [compliance_r, term_r, ltv_r], self.dissent_context
        )

        votes = [risk_vote, fraud_vote, policy_vote]

        # Run voting engine
        vote_result = self.voting_engine.run_vote(votes, self.episode)
        vote_result.all_sub_reports = all_sub_reports

        return vote_result

    def step(self, action: LoanAction, council_result: Optional[VoteResult] = None) -> Tuple[LoanObservation, float, bool, dict]:
        """
        Execute one step in the environment.
        Returns (next_obs, reward, done, info).
        done is always False (continuous environment).
        """
        obs = self.current_obs

        # ── MORTGAGE_MODE check (first, before anything) ──
        if obs.mortgage_value >= 1.3 * obs.loan_amount:
            reward = 1.0 if action == LoanAction.APPROVE else 0.0
            self._print_log(
                action, reward, obs,
                mode="mortgage_mode",
                council_result=None,
            )
            self.portfolio.update(action.value, reward)
            
            return (
                self.current_obs,
                reward,
                False,
                {
                    "mode": "mortgage_mode",
                    "vote_result": None,
                    "episode": self.episode,
                },
            )

        # ── EASY / MEDIUM (no council needed) ────────────
        if self.task_level in ["easy", "medium"]:
            reward_output = self.grader.grade(obs, action)
            reward = reward_output.score
            self._print_log(
                action, reward, obs,
                mode="normal",
                council_result=None,
            )
            self.portfolio.update(action.value, reward)
            
            return (
                self.current_obs,
                reward,
                False,
                {
                    "mode": "normal",
                    "episode": self.episode,
                    "reward_output": reward_output,
                },
            )

        # ── HARD: Full pipeline ──────────────────────────
        if council_result is None:
            council_result = self.get_council_recommendation()

        financial_risk = self._get_financial_risk(obs)
        behavioral_risk = self._get_behavioral_risk(obs)

        reward_output = self.grader.grade(
            obs, action, financial_risk, behavioral_risk
        )
        reward = calculate_shaped_reward(
            obs,
            action.value,
            self.task_level,
        )

        self._print_log(
            action, reward, obs,
            mode="normal",
            council_result=None,
        )
        self.portfolio.update(action.value, reward)
        
        return (
            self.current_obs,
            reward,
            False,
            {
                "mode": "normal",
                "episode": self.episode,
                "reward_output": reward_output,
                "grader_score": reward_output.score,
                "shaped_reward": reward,
                "financial_risk": financial_risk,
                "behavioral_risk": behavioral_risk,
            },
        )

    def state(self) -> LoanObservation:
        """Return current observation."""
        return self.current_obs

    def _get_financial_risk(self, obs: LoanObservation) -> str:
        """Classify financial risk level."""
        rs = (
            0.4 * (obs.credit_score - 300) / 550
            + 0.3 * (obs.income - 30000) / 170000
            + 0.2 * obs.employment_years / 40
            + 0.1
            * min(obs.mortgage_value / max(obs.loan_amount, 1), 2)
            / 2
        )
        if rs > 0.65:
            return "LOW"
        elif rs > 0.4:
            return "MODERATE"
        else:
            return "HIGH"

    def _get_behavioral_risk(self, obs: LoanObservation) -> str:
        """Classify behavioral risk level."""
        fs = _get_fraud_score(obs)
        if fs >= 2 or obs.application_consistency < 0.3:
            return "HIGH"
        elif fs == 1 or obs.application_consistency < 0.6:
            return "MODERATE"
        return "LOW"

    def _load_dissent_context(self) -> str:
        """Load last 5 dissent entries as context string."""
        path = "logs/dissent_log.json"
        if not os.path.exists(path):
            return "No dissent history yet."
        try:
            with open(path, "r") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, IOError):
            return "No dissent history yet."

        if not logs:
            return "No dissent history yet."

        recent = logs[-5:]
        lines = []
        for entry in recent:
            lines.append(
                f"Episode {entry['episode']}: "
                f"{entry['dissenting_agent']} dissented. "
                f"Voted {entry['minority_decision']} "
                f"while majority voted "
                f"{entry['majority_decision']}. "
                f"Reason: {entry['dissent_reason']}"
            )
        return "\n".join(lines)

    def _print_log(
        self,
        action: LoanAction,
        reward: float,
        obs: LoanObservation,
        mode: str,
        council_result: Optional[VoteResult],
    ) -> None:
        """Print formatted episode log to stdout."""
        print(f"\n[START]")
        print(f"Episode: {self.episode}")
        print(f"Task: {self.task_level}")
        print(f"Mode: {mode}")
        print(f"{'=' * 60}")

        if mode == "mortgage_mode":
            print(f"[MORTGAGE_MODE] Auto-approved.")
            print(
                f"mortgage={obs.mortgage_value:.0f} "
                f">= 1.3*loan={1.3 * obs.loan_amount:.0f}"
            )

        if council_result and council_result.dissent_log:
            d = council_result.dissent_log
            print(f"\n[DISSENT]")
            print(f"  Agent: {d.dissenting_agent}")
            print(
                f"  Voted: {d.minority_decision} "
                f"(majority: {d.majority_decision})"
            )
            print(f"  Reason: {d.dissent_reason}")

        print(f"\n[STEP]")
        print(f"  Final Engine Decision: {action.value}")
        print(f"  Reward: {reward:.4f}")
        print(f"[END] Episode {self.episode} Complete.")
        print(f"{'=' * 60}")
