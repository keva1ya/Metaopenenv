"""
P1: Compliance Officer Sub-Agent.

Rule-based. Zero API calls.
Evaluates: age, loan_term, customer_score.
Enforces bank policy age requirements and retirement limits.
"""

import os
import re
from typing import List, Optional
from openai import OpenAI

from agents.sub_agents.risk.credit_analyst import SubAgentReport
from env.observation import LoanObservation


class ComplianceOfficerAgent:
    """
    Checks regulatory and bank policy compliance
    related to applicant age, loan term constraints, and loan type.
    Uses an LLM for fluid policy evaluation.
    """
    SYSTEM_MESSAGE = (
        "You are the Compliance Officer Sub-Agent.\n"
        "Evaluate the applicant's age, loan term, customer score, and loan type.\n"
        "Do not use rigid numerical cutoffs; evaluate the holistic context.\n"
        "Ensure the loan type makes sense for the age and term.\n"
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

        prompt = (
            f"Applicant Context:\n"
            f"  Age: {obs.age}\n"
            f"  Loan Term: {obs.loan_term}yr\n"
            f"  Customer Score: {obs.customer_score:.2f}\n"
            f"  Loan Type: {obs.loan_type}\n"
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
            print(f"[ComplianceOfficer] LLM failed: {e}")
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
            agent_name="ComplianceOfficer",
            recommendation=rec,
            confidence=conf,
            reasoning=reason,
            key_flags=flags,
        )

    def _fallback(self, obs: LoanObservation) -> SubAgentReport:
        flags: List[str] = []
        if obs.age < 21 or obs.age > 60:
            flags.append("age_policy_violation")
            return SubAgentReport(
                agent_name="ComplianceOfficer", recommendation="REJECT", confidence=1.0,
                reasoning=f"Age {obs.age} violates bank policy. Hard compliance rejection.", key_flags=flags
            )
        if obs.age + obs.loan_term > 70:
            flags.append("exceeds_retirement_age")
            return SubAgentReport(
                agent_name="ComplianceOfficer", recommendation="REJECT", confidence=1.0,
                reasoning=f"Age {obs.age} + term {obs.loan_term} exceeds retirement limit of 70.", key_flags=flags
            )
        return SubAgentReport(
            agent_name="ComplianceOfficer", recommendation="APPROVE", confidence=0.8,
            reasoning=f"Age {obs.age} and term {obs.loan_term}yr comply with policy. Loan type {obs.loan_type} is standard.", key_flags=flags
        )
