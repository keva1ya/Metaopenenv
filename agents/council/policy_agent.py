"""
Policy Agent — Head Agent (Level 2).

Chief Policy Officer. Uses GPT-4o-mini.
Consumes: ComplianceOfficer + TermValidator + LTVChecker reports.
Prompt receives: compliance fields ONLY.

Hard rules enforced:
  P1: age 21-60
  P2: age + term <= 70
  P3: term <= 25 years
  P4: (debt + loan) / (income * term) < threshold
  P5: loan <= 10x annual income

On ANY exception: falls back to direct rule evaluation.
"""

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
    Synthesizes reports from 3 policy sub-agents using GPT-4o-mini.
    Falls back to direct rule evaluation on any failure.
    """

    SYSTEM_MESSAGE = (
        "You are the Chief Policy Officer.\n"
        "Check only bank policy compliance.\n"
        "Hard rules:\n"
        "  P1: age 21-60\n"
        "  P2: age+term <= 70\n"
        "  P3: term <= 25 years\n"
        "  P4: (debt+loan)/(income*term) < 0.5 "
        "(0.6 if customer_score >= 0.8)\n"
        "  P5: loan <= 10x annual income"
    )

    def __init__(self, client: Optional[OpenAI] = None, model_name: str = "minimax/minimax-m2.5:free"):
        self.client = client
        self.model_name = model_name

    def evaluate(
        self,
        obs: LoanObservation,
        sub_reports: List[SubAgentReport],
        dissent_context: str,
    ) -> HeadAgentVote:
        """
        Synthesize policy sub-agent reports into a single vote.
        Uses GPT-4o-mini with direct rule-check fallback.
        """
        try:
            if self.client is None:
                raise ValueError("No OpenRouter client configured")

            report_lines = []
            for r in sub_reports:
                report_lines.append(
                    f"{r.agent_name.upper()}: rec={r.recommendation} "
                    f"conf={r.confidence:.2f} flags={r.key_flags}"
                )

            user_prompt = (
                f"Sub-agent reports:\n"
                f"{'\n'.join(report_lines)}\n\n"
                f"Compliance context:\n"
                f"  Age: {obs.age}\n"
                f"  Loan Term: {obs.loan_term}yr\n"
                f"  Loan Amount: {obs.loan_amount:.0f}\n"
                f"  Income: {obs.income:.0f}\n"
                f"  Existing Debt: {obs.existing_debt:.0f}\n"
                f"  Mortgage Value: {obs.mortgage_value:.0f}\n"
                f"  Customer Score: {obs.customer_score:.2f}\n\n"
                f"DISSENT HISTORY: {dissent_context}\n\n"
                f"Respond ONLY:\n"
                f"DECISION: <APPROVE|REJECT|REQUEST_MORE_INFO>\n"
                f"CONFIDENCE: <0.0-1.0>\n"
                f"REASONING: <2 sentences>"
            )

            response = self.client.chat.completions.create(
                model=self.model_name,
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

        except Exception:
            return self._fallback(obs, sub_reports)

    def _parse_response(self, text: str):
        """Parse the structured LLM response."""
        decision = LoanAction.REJECT
        confidence = 0.5
        reasoning = text

        dec_match = re.search(
            r"DECISION[^\w]*(APPROVE|REJECT|REQUEST_MORE_INFO)", text, re.IGNORECASE
        )
        if dec_match:
            decision = LoanAction(dec_match.group(1).upper())

        conf_match = re.search(r"CONFIDENCE[^\w]*([0-9.]+)(%?)", text, re.IGNORECASE)
        if conf_match:
            val = float(conf_match.group(1))
            is_percent = conf_match.group(2) == "%"
            if is_percent or val > 1.0:
                val = val / 100.0
            confidence = max(0.0, min(1.0, val))

        reason_match = re.search(r"REASONING[^\w]*(.+)", text, re.IGNORECASE | re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        return decision, confidence, reasoning

    def _fallback(
        self, obs: LoanObservation, sub_reports: List[SubAgentReport]
    ) -> HeadAgentVote:
        """
        Direct rule evaluation fallback.
        Runs all 5 policy rules in pure Python.
        Any violation → REJECT. All pass → APPROVE.
        """
        violations: List[str] = []

        # P1: age 21-60
        if obs.age < 21 or obs.age > 60:
            violations.append(f"P1: age {obs.age} outside 21-60")

        # P2: age + term <= 70
        if obs.age + obs.loan_term > 70:
            violations.append(
                f"P2: age+term={obs.age + obs.loan_term} exceeds 70"
            )

        # P3: term <= 25
        if obs.loan_term > 25:
            violations.append(f"P3: term {obs.loan_term} exceeds 25yr")

        # P4: (debt + loan) / (income * term) < threshold
        threshold = 0.6 if obs.customer_score >= 0.8 else 0.5
        ratio = (obs.existing_debt + obs.loan_amount) / max(
            obs.income * obs.loan_term, 1
        )
        if ratio >= threshold:
            violations.append(
                f"P4: debt_ratio {ratio:.2f} >= {threshold}"
            )

        # P5: loan <= 10x income
        if obs.loan_amount > 10 * obs.income:
            violations.append(
                f"P5: loan {obs.loan_amount:.0f} > 10x income {obs.income:.0f}"
            )

        if violations:
            return HeadAgentVote(
                agent_name="PolicyAgent",
                decision=LoanAction.REJECT,
                reasoning=f"Fallback: Policy violations: {'; '.join(violations)}.",
                confidence=0.95,
                sub_agent_reports=sub_reports,
            )

        return HeadAgentVote(
            agent_name="PolicyAgent",
            decision=LoanAction.APPROVE,
            reasoning="Fallback: All 5 policy rules pass. Compliant application.",
            confidence=0.9,
            sub_agent_reports=sub_reports,
        )
