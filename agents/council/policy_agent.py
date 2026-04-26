"""
Policy Agent - Head Agent (Level 2).

Chief Policy Officer. Uses GPT-style synthesis through OpenRouter.
Consumes: ComplianceOfficer + TermValidator + LTVChecker reports.
Prompt receives broader application context.

This agent uses policy judgment rather than rigid hard-coded
gates. On any exception, it falls back to a soft synthesis
of sub-agent signals instead of deterministic rejection rules.
"""

import os
import re
from typing import List, Optional

from openai import OpenAI

from agents.council.risk_agent import HeadAgentVote
from agents.sub_agents.risk.credit_analyst import SubAgentReport
from env.action import LoanAction
from env.observation import LoanObservation


class PolicyAgent:
    """
    Chief Policy Officer head agent.
    Synthesizes reports from 3 policy sub-agents using an LLM.
    Falls back to a soft aggregation policy judgment on failure.
    """

    SYSTEM_MESSAGE = (
        "You are the Chief Policy Officer.\n"
        "Evaluate policy suitability using holistic judgment.\n"
        "Do not apply rigid hard-coded cutoffs as mandatory vetoes.\n"
        "Weigh the policy sub-agent reports together with the full "
        "application context, including affordability, collateral, "
        "customer quality, term structure, debt burden, and edge cases.\n"
        "Use REQUEST_MORE_INFO when the application looks policy-uncertain "
        "rather than clearly acceptable or clearly unsuitable."
    )

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model_name: str = "openrouter/auto",
    ):
        self.client = client
        self.model_name = model_name

    def _complete_with_fallback(self, messages: list, temperature: float, max_tokens: int):
        """Try primary model, then fallback model if configured."""
        fallback_model = os.getenv("OPENROUTER_FALLBACK_MODEL", "openrouter/auto")
        candidates = [self.model_name]
        if fallback_model and fallback_model not in candidates:
            candidates.append(fallback_model)

        last_exc = None
        for model in candidates:
            try:
                return self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:
                last_exc = exc
                continue
        raise last_exc if last_exc is not None else RuntimeError("No model candidates configured")

    def evaluate(
        self,
        obs: LoanObservation,
        sub_reports: List[SubAgentReport],
        dissent_context: str,
    ) -> HeadAgentVote:
        """
        Synthesize policy sub-agent reports into a single vote.
        Uses the configured LLM with soft aggregation fallback.
        """
        try:
            if self.client is None:
                raise ValueError("No OpenRouter client configured")

            report_lines = []
            for report in sub_reports:
                report_lines.append(
                    f"{report.agent_name.upper()}: rec={report.recommendation} "
                    f"conf={report.confidence:.2f} flags={report.key_flags}"
                )

            user_prompt = (
                f"Sub-agent reports:\n"
                f"{'\n'.join(report_lines)}\n\n"
                f"Application context:\n"
                f"  Age: {obs.age}\n"
                f"  Loan Term: {obs.loan_term}yr\n"
                f"  Loan Amount: {obs.loan_amount:.0f}\n"
                f"  Income: {obs.income:.0f}\n"
                f"  Existing Debt: {obs.existing_debt:.0f}\n"
                f"  Mortgage Value: {obs.mortgage_value:.0f}\n"
                f"  Employment Years: {obs.employment_years}\n"
                f"  Monthly Expenses: {obs.monthly_expenses:.0f}\n"
                f"  Dependents: {obs.num_dependents}\n"
                f"  Previous Defaults: {obs.previous_defaults}\n"
                f"  GST Registered: {obs.gst_registered}\n"
                f"  ITR Filed Years: {obs.itr_filed_years}\n"
                f"  Customer Score: {obs.customer_score:.2f}\n"
                f"  Loan Type: {obs.loan_type}\n\n"
                f"DISSENT HISTORY: {dissent_context}\n\n"
                f"Respond ONLY:\n"
                f"DECISION: <APPROVE|REJECT|REQUEST_MORE_INFO>\n"
                f"CONFIDENCE: <0.0-1.0>\n"
                f"REASONING: <2 sentences>"
            )

            response = self._complete_with_fallback(
                messages=[
                    {"role": "system", "content": self.SYSTEM_MESSAGE},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            )

            text = response.choices[0].message.content.strip()
            decision, confidence, reasoning = self._parse_response(text)

            return HeadAgentVote(
                agent_name="PolicyAgent",
                decision=decision,
                reasoning=reasoning,
                confidence=confidence,
                sub_agent_reports=sub_reports,
            )

        except Exception as exc:
            print(f"[PolicyAgent] LLM call failed, using fallback: {exc}")
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
