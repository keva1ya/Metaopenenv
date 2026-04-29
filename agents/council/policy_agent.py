"""
Policy Agent - Head Agent (Level 2).

Chief Policy Officer. Uses HuggingFace Inference API with Keva1ya/mistral-risk-model.
Consumes: ComplianceOfficer + TermValidator + LTVChecker reports.
Prompt receives broader application context.

This agent uses policy judgment rather than rigid hard-coded
gates. On any exception, it falls back to a soft synthesis
of sub-agent signals instead of deterministic rejection rules.
"""

import re
import requests
import os
from typing import List, Optional
from huggingface_hub import InferenceClient

from agents.council.risk_agent import HeadAgentVote
from agents.sub_agents.risk.credit_analyst import SubAgentReport
from env.action import LoanAction
from env.observation import LoanObservation


class PolicyAgent:
    """
    Chief Policy Officer head agent.
    Synthesizes reports from 3 policy sub-agents using Keva1ya/mistral-risk-model.
    Falls back to a soft aggregation policy judgment on failure.
    """

    def __init__(self, client=None, model_name=None):
        pass

    def evaluate(
        self,
        obs: LoanObservation,
        sub_reports: List[SubAgentReport],
        dissent_context: str,
    ) -> HeadAgentVote:
        """
        Synthesize policy sub-agent reports into a single vote.
        Uses the configured HuggingFace model with soft aggregation fallback.
        """
        try:
            token = os.getenv("HF_TOKEN", "")
            client = InferenceClient(
                model="Keva1ya/mistral-risk-model",
                token=token
            )

            report_lines = []
            for report in sub_reports:
                report_lines.append(
                    f"{report.agent_name.upper()}: rec={report.recommendation} "
                    f"conf={report.confidence:.2f} flags={report.key_flags}"
                )

            prompt_content = (
                f"Evaluate policy suitability using holistic judgment.\n"
                f"Weigh the policy sub-agent reports together with the full applicant context.\n\n"
                f"Sub-Agent Reports:\n{chr(10).join(report_lines)}\n\n"
                f"Applicant Data: Age {obs.age}, Loan Term {obs.loan_term}yr, "
                f"Loan Amount {obs.loan_amount:.0f}, Income {obs.income:.0f}, "
                f"Existing Debt {obs.existing_debt:.0f}, Mortgage Value {obs.mortgage_value:.0f}, "
                f"Employment Years {obs.employment_years}, Monthly Expenses {obs.monthly_expenses:.0f}, "
                f"Dependents {obs.num_dependents}, Previous Defaults {obs.previous_defaults}, "
                f"GST Registered {obs.gst_registered}, ITR Filed Years {obs.itr_filed_years}, "
                f"Customer Score {obs.customer_score:.2f}, Loan Type {obs.loan_type}\n\n"
                f"Respond ONLY with:\nDECISION: <APPROVE|REJECT|REQUEST_MORE_INFO>\nCONFIDENCE: <0.0-1.0>\nREASONING: <2 sentences>"
            )

            # Mistral commonly uses a different chat template than Qwen, e.g. [INST] [/INST]
            text = client.text_generation(
                prompt=f"[INST] {prompt_content} [/INST]\n",
                max_new_tokens=200,
                temperature=0.3
            )

            decision, confidence, reasoning = self._parse_response(text)

            return HeadAgentVote(
                agent_name="PolicyAgent",
                decision=decision,
                reasoning=reasoning,
                confidence=confidence,
                sub_agent_reports=sub_reports,
            )

        except Exception as exc:
            print(f"[PolicyAgent] HuggingFace API call failed, using fallback: {exc}")
            return self._fallback(sub_reports)

    def _parse_response(self, text: str):
        """Parse the structured LLM response."""
        decision = LoanAction.REJECT
        confidence = 0.5
        reasoning = text

        dec_match = re.search(
            r"DECISION[^\w]*(APPROVE|REJECT|REQUEST_MORE_INFO)",
            text,
            re.IGNORECASE,
        )
        if dec_match:
            decision = LoanAction(dec_match.group(1).upper())

        conf_match = re.search(
            r"CONFIDENCE[^\w]*([0-9.]+)(%?)",
            text,
            re.IGNORECASE,
        )
        if conf_match:
            value = float(conf_match.group(1))
            is_percent = conf_match.group(2) == "%"
            if is_percent or value > 1.0:
                value = value / 100.0
            confidence = max(0.0, min(1.0, value))

        reason_match = re.search(
            r"REASONING[^\w]*(.+)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if reason_match:
            reasoning = reason_match.group(1).strip()

        return decision, confidence, reasoning

    def _fallback(self, sub_reports: List[SubAgentReport]) -> HeadAgentVote:
        """
        Soft policy aggregation fallback.
        Uses confidence-weighted sub-agent synthesis rather than
        deterministic hard-rule enforcement.
        """
        score_map = {
            "APPROVE": 1.0,
            "REVIEW": 0.0,
            "REJECT": -1.0,
        }
        total_weight = sum(max(report.confidence, 0.0) for report in sub_reports)
        weighted_score = sum(
            score_map.get(report.recommendation, 0.0)
            * max(report.confidence, 0.0)
            for report in sub_reports
        )

        if total_weight <= 1e-8:
            decision = LoanAction.REQUEST_MORE_INFO
            confidence = 0.5
            reasoning = (
                "Fallback: Policy evidence is insufficient, so requesting more "
                "information instead of applying a rigid rule-based decision."
            )
        else:
            normalized_score = weighted_score / total_weight
            avg_confidence = total_weight / max(len(sub_reports), 1)

            if normalized_score >= 0.35:
                decision = LoanAction.APPROVE
            elif normalized_score <= -0.35:
                decision = LoanAction.REJECT
            else:
                decision = LoanAction.REQUEST_MORE_INFO

            confidence = max(0.45, min(0.9, avg_confidence))
            reasoning = (
                "Fallback: Soft policy synthesis of sub-agent signals produced "
                f"a normalized score of {normalized_score:.2f}, leading to "
                f"{decision.value} without rigid hard-rule enforcement."
            )

        return HeadAgentVote(
            agent_name="PolicyAgent",
            decision=decision,
            reasoning=reasoning,
            confidence=confidence,
            sub_agent_reports=sub_reports,
        )
