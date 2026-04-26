"""
P3: LTV Checker Sub-Agent.

Rule-based. Zero API calls.
Evaluates: mortgage_value, loan_amount, existing_debt.
Checks Loan-to-Value ratio for secured loans.
"""

import os
import re
from typing import List, Optional
from openai import OpenAI

from agents.sub_agents.risk.credit_analyst import SubAgentReport
from env.observation import LoanObservation


class LTVCheckerAgent:
    """
    Evaluates Loan-to-Value (LTV) ratio to assess
    collateral adequacy for the requested loan.
    Uses an LLM for fluid policy evaluation.
    """
    SYSTEM_MESSAGE = (
        "You are the LTV Checker Sub-Agent.\n"
        "Evaluate the Loan-to-Value (LTV) ratio (loan_amount / mortgage_value).\n"
        "Do not use rigid numerical cutoffs; evaluate the holistic context.\n"
        "Consider the loan_type when assessing if the LTV is risky.\n"
        "Respond ONLY in this format:\n"
        "RECOMMENDATION: <APPROVE|REJECT|REVIEW>\n"
        "CONFIDENCE: <0.0-1.0>\n"
        "REASONING: <2-3 sentences>\n"
        "FLAGS: <comma-separated flags>"
    )

    def __init__(self, client: Optional[OpenAI] = None, model_name: str = "openrouter/auto"):
        self.client = client
        self.model_name = model_name

    def analyze(self, obs: LoanObservation) -> SubAgentReport:
        if self.client is None:
            return self._fallback(obs)

        ltv = obs.loan_amount / max(obs.mortgage_value, 1) if obs.mortgage_value > 0 else float('inf')
        
        prompt = (
            f"Applicant Context:\n"
            f"  Loan Amount: {obs.loan_amount:.0f}\n"
            f"  Mortgage Value (Collateral): {obs.mortgage_value:.0f}\n"
            f"  Calculated LTV: {ltv:.2f}\n"
            f"  Loan Type: {obs.loan_type}\n"
            f"  Existing Debt: {obs.existing_debt:.0f}\n"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=150,
            )
            text = response.choices[0].message.content.strip()
            return self._parse_response(text)
        except Exception as e:
            print(f"[LTVChecker] LLM failed: {e}")
            return self._fallback(obs)

    def _parse_response(self, text: str) -> SubAgentReport:
        rec_match = re.search(r"RECOMMENDATION[^\w]*(APPROVE|REJECT|REVIEW)", text, re.IGNORECASE)
        conf_match = re.search(r"CONFIDENCE[^\w]*([0-9.]+)", text, re.IGNORECASE)
        reason_match = re.search(r"REASONING[^\w]*(.*?)(?=FLAGS|$)", text, re.IGNORECASE | re.DOTALL)
        flags_match = re.search(r"FLAGS[^\w]*(.+)", text, re.IGNORECASE)

        rec = rec_match.group(1).upper() if rec_match else "REVIEW"
        conf = min(1.0, max(0.0, float(conf_match.group(1)))) if conf_match else 0.5
        reason = reason_match.group(1).strip() if reason_match else text
        flags = [f.strip() for f in flags_match.group(1).split(",")] if flags_match else []

        return SubAgentReport(
            agent_name="LTVChecker",
            recommendation=rec,
            confidence=conf,
            reasoning=reason,
            key_flags=flags,
        )

    def _fallback(self, obs: LoanObservation) -> SubAgentReport:
        flags: List[str] = []
        if obs.mortgage_value == 0:
            flags.append("unsecured_no_collateral")
            return SubAgentReport(
                agent_name="LTVChecker", recommendation="REVIEW", confidence=0.7,
                reasoning=f"No collateral provided. Loan is unsecured.", key_flags=flags
            )
        ltv = obs.loan_amount / obs.mortgage_value
        if ltv > 0.9:
            flags.append(f"ltv_{ltv:.2f}")
            return SubAgentReport(
                agent_name="LTVChecker", recommendation="REJECT", confidence=0.85,
                reasoning=f"LTV ratio {ltv:.2f} exceeds 90% threshold.", key_flags=flags
            )
        flags.append(f"ltv_{ltv:.2f}")
        return SubAgentReport(
            agent_name="LTVChecker", recommendation="APPROVE", confidence=0.9,
            reasoning=f"LTV ratio {ltv:.2f} is acceptable.", key_flags=flags
        )
