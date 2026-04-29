"""
P2: Term Validator Sub-Agent.

Rule-based. Zero API calls.
Evaluates: loan_term, loan_amount, income.
Validates loan term reasonableness and EMI affordability.
"""

import os
import re
from typing import List, Optional
from huggingface_hub import InferenceClient

from agents.sub_agents.risk.credit_analyst import SubAgentReport
from env.observation import LoanObservation


class TermValidatorAgent:
    """
    Validates that the loan term is reasonable given
    the loan amount and applicant income.
    Uses an LLM for fluid policy evaluation.
    """
    SYSTEM_MESSAGE = (
        "You are the Term Validator Sub-Agent.\n"
        "Evaluate the loan term in relation to the loan amount, income, and loan type.\n"
        "Do not use rigid numerical cutoffs; evaluate the holistic context.\n"
        "Consider if the resulting EMI is affordable and the term is reasonable for the loan type.\n"
        "Respond ONLY in this format:\n"
        "RECOMMENDATION: <APPROVE|REJECT|REVIEW>\n"
        "CONFIDENCE: <0.0-1.0>\n"
        "REASONING: <2-3 sentences>\n"
        "FLAGS: <comma-separated flags>"
    )

    def __init__(self, model_name: str = "Keva1ya/mistral-risk-model"):
        self.token = os.getenv("HF_TOKEN")
        self.model_name = model_name
        self.client = InferenceClient(model=self.model_name, token=self.token) if self.token else None

    def analyze(self, obs: LoanObservation) -> SubAgentReport:
        if self.client is None:
            return self._fallback(obs)

        monthly_install = obs.loan_amount / max((obs.loan_term * 12), 1)
        monthly_income = obs.income / 12
        emi_ratio = monthly_install / max(monthly_income, 1)
        
        prompt = (
            f"Applicant Context:\n"
            f"  Loan Amount: {obs.loan_amount:.0f}\n"
            f"  Loan Term: {obs.loan_term}yr\n"
            f"  Monthly Income: {monthly_income:.0f}\n"
            f"  Estimated EMI: {monthly_install:.0f} ({emi_ratio*100:.1f}% of income)\n"
            f"  Loan Type: {obs.loan_type}\n"
        )
        
        try:
            text = self.client.text_generation(
                prompt=f"[INST] {self.SYSTEM_MESSAGE}\n\n{prompt} [/INST]\n",
                max_new_tokens=150,
                temperature=0.3,
            )
            return self._parse_response(text)
        except Exception as e:
            print(f"[TermValidator] LLM failed: {e}")
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
            agent_name="TermValidator",
            recommendation=rec,
            confidence=conf,
            reasoning=reason,
            key_flags=flags,
        )

    def _fallback(self, obs: LoanObservation) -> SubAgentReport:
        flags: List[str] = []
        if obs.loan_term > 25:
            flags.append("term_exceeds_25yr_limit")
            return SubAgentReport(
                agent_name="TermValidator", recommendation="REJECT", confidence=1.0,
                reasoning=f"Loan term {obs.loan_term}yr exceeds maximum 25-year limit.", key_flags=flags
            )
        monthly_install = obs.loan_amount / (obs.loan_term * 12)
        monthly_income = obs.income / 12
        if monthly_install > 0.5 * monthly_income:
            flags.append("emi_gt_50pct_income")
            return SubAgentReport(
                agent_name="TermValidator", recommendation="REVIEW", confidence=0.65,
                reasoning=f"EMI {monthly_install:.0f} exceeds 50% of monthly income {monthly_income:.0f}.", key_flags=flags
            )
        flags.append(f"emi_pct_{monthly_install / max(monthly_income, 1) * 100:.1f}")
        return SubAgentReport(
            agent_name="TermValidator", recommendation="APPROVE", confidence=0.85,
            reasoning=f"Loan term {obs.loan_term}yr is reasonable for EMI.", key_flags=flags
        )
