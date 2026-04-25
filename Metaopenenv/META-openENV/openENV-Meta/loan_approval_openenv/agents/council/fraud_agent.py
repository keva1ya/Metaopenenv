"""
Fraud Agent — Head Agent (Level 2).

Chief Fraud Officer. Uses GPT-4o-mini.
Consumes: PatternChecker + ConsistencyChecker reports.
Prompt receives: behavioral fields ONLY.
Income/employment only used for inconsistency context.

Strong fraud signal (fraud_score>=2 or consistency<0.3)
forces a REJECT vote.

On ANY exception: falls back to rule-based decision.
"""

import re
from typing import List, Optional

from openai import OpenAI

from agents.council.risk_agent import HeadAgentVote
from agents.sub_agents.risk.credit_analyst import SubAgentReport
from env.action import LoanAction
from env.observation import LoanObservation


class FraudAgent:
    """
    Chief Fraud Officer head agent.
    Synthesizes reports from 2 fraud sub-agents using GPT-4o-mini.
    Falls back to rule-based scoring on any failure.
    """

    SYSTEM_MESSAGE = (
        "You are the Chief Fraud Officer.\n"
        "Evaluate ONLY behavioral and fraud signals.\n"
        "Never evaluate financial capacity.\n"
        "Strong fraud signal (fraud_score>=2 or consistency<0.3) "
        "means you MUST vote REJECT."
    )

    def __init__(self, client: Optional[OpenAI] = None):
        self.client = client

    def evaluate(
        self,
        obs: LoanObservation,
        sub_reports: List[SubAgentReport],
        dissent_context: str,
    ) -> HeadAgentVote:
        """
        Synthesize fraud sub-agent reports into a single vote.
        Uses GPT-4o-mini with rule-based fallback.
        """
        try:
            if self.client is None:
                raise ValueError("No OpenAI client configured")

            report_lines = []
            for r in sub_reports:
                report_lines.append(
                    f"{r.agent_name.upper()}: rec={r.recommendation} "
                    f"conf={r.confidence:.2f} flags={r.key_flags}"
                )

            # Behavioral fields only (income/employment only for context)
            user_prompt = (
                f"Sub-agent reports:\n"
                f"{'\n'.join(report_lines)}\n\n"
                f"Behavioral context:\n"
                f"  Spending Volatility: {obs.spending_volatility:.2f}\n"
                f"  Loan Purpose Clarity: {obs.loan_purpose_clarity:.2f}\n"
                f"  Application Consistency: {obs.application_consistency:.2f}\n"
                f"  Income (context only): {obs.income:.0f}\n"
                f"  Employment (context only): {obs.employment_years}yr\n\n"
                f"DISSENT HISTORY: {dissent_context}\n\n"
                f"Respond ONLY:\n"
                f"DECISION: <APPROVE|REJECT|REQUEST_MORE_INFO>\n"
                f"CONFIDENCE: <0.0-1.0>\n"
                f"REASONING: <2 sentences>"
            )

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
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
                agent_name="FraudAgent",
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
            r"DECISION:\s*(APPROVE|REJECT|REQUEST_MORE_INFO)", text, re.IGNORECASE
        )
        if dec_match:
            decision = LoanAction(dec_match.group(1).upper())

        conf_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", text)
        if conf_match:
            confidence = max(0.0, min(1.0, float(conf_match.group(1))))

        reason_match = re.search(r"REASONING:\s*(.+)", text, re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        return decision, confidence, reasoning

    def _fallback(
        self, obs: LoanObservation, sub_reports: List[SubAgentReport]
    ) -> HeadAgentVote:
        """Rule-based fallback when LLM call fails."""
        # Extract fraud_score from pattern checker flags
        fraud_score = 0
        for r in sub_reports:
            if r.agent_name == "PatternChecker":
                for flag in r.key_flags:
                    if "fraud_score:" in flag:
                        try:
                            fraud_score = int(
                                flag.split(":")[1].split("/")[0].strip()
                            )
                        except (ValueError, IndexError):
                            pass

        consistency = obs.application_consistency

        if fraud_score >= 2 or consistency < 0.3:
            decision = LoanAction.REJECT
            confidence = 0.9
            reasoning = (
                f"Fallback: Strong fraud signal. "
                f"fraud_score={fraud_score}, consistency={consistency:.2f}."
            )
        elif fraud_score == 1:
            decision = LoanAction.REQUEST_MORE_INFO
            confidence = 0.65
            reasoning = (
                f"Fallback: Moderate fraud signal. "
                f"fraud_score={fraud_score}. Requesting more info."
            )
        else:
            decision = LoanAction.APPROVE
            confidence = 0.8
            reasoning = (
                f"Fallback: No significant fraud signals. "
                f"fraud_score={fraud_score}, consistency={consistency:.2f}."
            )

        return HeadAgentVote(
            agent_name="FraudAgent",
            decision=decision,
            reasoning=reasoning,
            confidence=confidence,
            sub_agent_reports=sub_reports,
        )
