"""
Risk Agent — Head Agent (Level 2).

Chief Risk Officer. Uses HuggingFace Inference API with keva1ya/qwen-risk-merged.
Consumes: CreditAnalyst + IncomeVerifier + DebtAnalyzer reports.
Prompt receives: financial fields ONLY (no behavioral).

On ANY exception: falls back to majority vote of sub-agents.
"""

import re
import os
from typing import List, Optional
from huggingface_hub import InferenceClient

from agents.sub_agents.risk.credit_analyst import SubAgentReport
from env.action import LoanAction
from env.observation import LoanObservation


class HeadAgentVote:
    """
    Structured vote from a head agent.
    Imported here to avoid circular imports — used as a
    simple container class.
    """

    def __init__(
        self,
        agent_name: str,
        decision: LoanAction,
        reasoning: str,
        confidence: float,
        sub_agent_reports: List[SubAgentReport],
    ):
        self.agent_name = agent_name
        self.decision = decision
        self.reasoning = reasoning
        self.confidence = confidence
        self.sub_agent_reports = sub_agent_reports


class RiskAgent:
    """
    Chief Risk Officer head agent.
    Synthesizes reports from 3 risk sub-agents using keva1ya/qwen-risk-merged.
    Falls back to majority vote on any failure.
    """

    def __init__(self, client=None, model_name=None):
        # We ignore client and model_name to use the direct HF Inference API
        pass

    def evaluate(
        self,
        obs: LoanObservation,
        sub_reports: List[SubAgentReport],
        dissent_context: str,
    ) -> HeadAgentVote:
        """
        Synthesize sub-agent reports into a single risk vote using HuggingFace API.
        Falls back to majority rule.
        """
        try:
            token = os.getenv("HF_TOKEN", "")
            client = InferenceClient(
                model="Keva1ya/qwen-risk-merged",
                token=token
            )
            
            # Format sub-agent reports
            report_lines = []
            for r in sub_reports:
                report_lines.append(
                    f"{r.agent_name.upper()}: rec={r.recommendation} "
                    f"conf={r.confidence:.2f} flags={r.key_flags}"
                )

            prompt_content = (
                f"Assess credit risk strictly based on these sub-agent reports and financial data.\n\n"
                f"Sub-Agent Reports:\n{chr(10).join(report_lines)}\n\n"
                f"Financial Data: Income {obs.income:.0f}, Credit Score {obs.credit_score}, "
                f"Employment {obs.employment_years}yr, Loan Amount {obs.loan_amount:.0f}, "
                f"Existing Debt {obs.existing_debt:.0f}, Mortgage Value {obs.mortgage_value:.0f}, "
                f"Loan Term {obs.loan_term}yr.\n\n"
                f"Respond ONLY with:\nDECISION: <APPROVE|REJECT|REQUEST_MORE_INFO>\nCONFIDENCE: <0.0-1.0>\nREASONING: <2 sentences>"
            )

            # Qwen uses ChatML tags
            text = client.text_generation(
                prompt=f"<|im_start|>user\n{prompt_content}\n<|im_end|>\n<|im_start|>assistant\n",
                max_new_tokens=200,
                temperature=0.3
            )

            decision, confidence, reasoning = self._parse_response(text)

            return HeadAgentVote(
                agent_name="RiskAgent",
                decision=decision,
                reasoning=reasoning,
                confidence=confidence,
                sub_agent_reports=sub_reports,
            )

        except Exception as exc:
            print(f"[RiskAgent] HuggingFace API call failed, using fallback: {exc}")
            return self._fallback(sub_reports)

    def _parse_response(self, text: str):
        """Parse the structured LLM response."""
        decision = LoanAction.REJECT
        confidence = 0.5
        reasoning = text

        # Extract decision
        dec_match = re.search(
            r"DECISION[^\w]*(APPROVE|REJECT|REQUEST_MORE_INFO)", text, re.IGNORECASE
        )
        if dec_match:
            decision = LoanAction(dec_match.group(1).upper())

        # Extract confidence
        conf_match = re.search(r"CONFIDENCE[^\w]*([0-9.]+)(%?)", text, re.IGNORECASE)
        if conf_match:
            val = float(conf_match.group(1))
            is_percent = conf_match.group(2) == "%"
            if is_percent or val > 1.0:
                val = val / 100.0
            confidence = max(0.0, min(1.0, val))

        # Extract reasoning
        reason_match = re.search(r"REASONING[^\w]*(.+)", text, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        return decision, confidence, reasoning

    def _fallback(self, sub_reports: List[SubAgentReport]) -> HeadAgentVote:
        """Majority vote fallback when LLM call fails."""
        vote_map = {"APPROVE": 0, "REJECT": 0, "REQUEST_MORE_INFO": 0}
        total_conf = 0.0

        for r in sub_reports:
            rec = r.recommendation
            if rec == "REVIEW":
                rec = "REQUEST_MORE_INFO"
            if rec in vote_map:
                vote_map[rec] += 1
            total_conf += r.confidence

        majority = max(vote_map, key=vote_map.get)
        avg_conf = total_conf / max(len(sub_reports), 1)

        return HeadAgentVote(
            agent_name="RiskAgent",
            decision=LoanAction(majority),
            reasoning=f"Fallback majority vote: {vote_map}. Using rule-based aggregation.",
            confidence=avg_conf,
            sub_agent_reports=sub_reports,
        )
