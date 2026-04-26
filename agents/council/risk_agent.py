"""
Risk Agent — Head Agent (Level 2).

Chief Risk Officer. Uses GPT-4o-mini.
Consumes: CreditAnalyst + IncomeVerifier + DebtAnalyzer reports.
Prompt receives: financial fields ONLY (no behavioral).

On ANY exception: falls back to majority vote of sub-agents.
"""

import os
import re
from typing import List, Optional

from openai import OpenAI

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
    Synthesizes reports from 3 risk sub-agents using GPT-4o-mini.
    Falls back to majority vote on any failure.
    """

    SYSTEM_MESSAGE = (
        "You are the Chief Risk Officer.\n"
        "Evaluate ONLY financial risk.\n"
        "Never consider behavioral signals.\n"
        "Honest behavior does not guarantee financial safety."
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
        Synthesize sub-agent reports into a single risk vote.
        Uses GPT-4o-mini with fallback to majority rule.
        """
        try:
            if self.client is None:
                raise ValueError("No OpenRouter client configured")

            # Build user prompt with financial fields only
            report_lines = []
            for r in sub_reports:
                report_lines.append(
                    f"{r.agent_name.upper()}: rec={r.recommendation} "
                    f"conf={r.confidence:.2f} flags={r.key_flags}"
                )

            user_prompt = (
                f"<sub_agent_reports>\n"
                f"{'\n'.join(report_lines)}\n"
                f"</sub_agent_reports>\n\n"
                f"<applicant_data>\n"
                f"  Income: {obs.income:.0f}\n"
                f"  Credit Score: {obs.credit_score}\n"
                f"  Employment: {obs.employment_years}yr\n"
                f"  Loan Amount: {obs.loan_amount:.0f}\n"
                f"  Existing Debt: {obs.existing_debt:.0f}\n"
                f"  Mortgage Value: {obs.mortgage_value:.0f}\n"
                f"  Loan Term: {obs.loan_term}yr\n"
                f"</applicant_data>\n\n"
                f"<dissent_history>\n"
                f"{dissent_context}\n"
                f"</dissent_history>\n\n"
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
                agent_name="RiskAgent",
                decision=decision,
                reasoning=reasoning,
                confidence=confidence,
                sub_agent_reports=sub_reports,
            )

        except Exception as exc:
            print(f"[RiskAgent] LLM call failed, using fallback: {exc}")
            # Fallback: majority vote of sub-agents
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

        # Find majority
        majority = max(vote_map, key=vote_map.get)
        avg_conf = total_conf / max(len(sub_reports), 1)

        return HeadAgentVote(
            agent_name="RiskAgent",
            decision=LoanAction(majority),
            reasoning=f"Fallback majority vote: {vote_map}. Using rule-based aggregation.",
            confidence=avg_conf,
            sub_agent_reports=sub_reports,
        )
