"""
Fraud Agent - Head Agent (Level 2).

Chief Fraud Officer. Uses GPT-style synthesis through OpenRouter.
Consumes: PatternChecker + ConsistencyChecker + KYCVerifier reports.
Prompt receives behavioral fields plus limited context features.

Explicit safeguards:
- Bank KYC failed => hard REJECT veto
- Bank KYC pending + additional fraud signals => likely identity/fraud
  escalation and hard REJECT
- Bank KYC pending alone => REQUEST_MORE_INFO / human verification
"""

import os
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
    Synthesizes reports from fraud sub-agents using an LLM, with
    explicit post-check guards for identity-fraud veto behavior.
    """

    SYSTEM_MESSAGE = (
        "You are the Chief Fraud Officer.\n"
        "Evaluate only behavioral and fraud signals.\n"
        "Never evaluate financial capacity as credit risk.\n"
        "KYCVerifier reports the bank-owned KYC status.\n"
        "If bank KYC failed, you must vote REJECT.\n"
        "If bank KYC is pending but not failed, prefer "
        "REQUEST_MORE_INFO for human verification.\n"
        "If pending KYC appears together with other fraud indicators, "
        "treat it as likely identity/fraud escalation and vote REJECT."
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
        Synthesize fraud sub-agent reports into a single vote.
        Uses the configured LLM with deterministic guardrails and fallback.
        """
        identity_status = self._extract_identity_status(sub_reports)
        fraud_score = self._extract_pattern_fraud_score(sub_reports)
        consistency = obs.application_consistency
        consistency_flags = self._consistency_flag_count(sub_reports)

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
                f"<sub_agent_reports>\n"
                f"{'\n'.join(report_lines)}\n"
                f"</sub_agent_reports>\n\n"
                f"<applicant_data>\n"
                f"  Spending Volatility: {obs.spending_volatility:.2f}\n"
                f"  Loan Purpose Clarity: {obs.loan_purpose_clarity:.2f}\n"
                f"  Application Consistency: {obs.application_consistency:.2f}\n"
                f"  Previous Defaults: {obs.previous_defaults}\n"
                f"  Income (context only): {obs.income:.0f}\n"
                f"  Employment (context only): {obs.employment_years}yr\n"
                f"</applicant_data>\n\n"
                f"<identity_guidance>\n"
                f"  Bank KYC status: {identity_status}\n"
                f"  Pattern fraud score: {fraud_score}\n"
                f"  Consistency flag count: {consistency_flags}\n"
                f"</identity_guidance>\n\n"
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
                max_tokens=220,
            )

            text = response.choices[0].message.content.strip()
            decision, confidence, reasoning = self._parse_response(text)
            decision, confidence, reasoning = self._apply_identity_guards(
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                identity_status=identity_status,
                fraud_score=fraud_score,
                consistency=consistency,
                consistency_flags=consistency_flags,
            )

            return HeadAgentVote(
                agent_name="FraudAgent",
                decision=decision,
                reasoning=reasoning,
                confidence=confidence,
                sub_agent_reports=sub_reports,
            )

        except Exception as exc:
            print(f"[FraudAgent] LLM call failed, using fallback: {exc}")
            return self._fallback(
                obs=obs,
                sub_reports=sub_reports,
                identity_status=identity_status,
                fraud_score=fraud_score,
                consistency_flags=consistency_flags,
            )

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

    def _extract_pattern_fraud_score(self, sub_reports: List[SubAgentReport]) -> int:
        fraud_score = 0
        for report in sub_reports:
            if report.agent_name != "PatternChecker":
                continue
            for flag in report.key_flags:
                if "fraud_score:" not in flag:
                    continue
                try:
                    fraud_score = int(flag.split(":")[1].split("/")[0].strip())
                except (ValueError, IndexError):
                    fraud_score = 0
        return fraud_score

    def _extract_identity_status(self, sub_reports: List[SubAgentReport]) -> str:
        for report in sub_reports:
            if report.agent_name != "KYCVerifier":
                continue
            if report.recommendation == "REJECT":
                return "FAILED"
            if report.recommendation == "REVIEW":
                return "PENDING"
            return "VERIFIED"
        return "UNKNOWN"

    def _consistency_flag_count(self, sub_reports: List[SubAgentReport]) -> int:
        for report in sub_reports:
            if report.agent_name == "ConsistencyChecker":
                return len(report.key_flags)
        return 0

    def _apply_identity_guards(
        self,
        decision: LoanAction,
        confidence: float,
        reasoning: str,
        identity_status: str,
        fraud_score: int,
        consistency: float,
        consistency_flags: int,
    ):
        extra_fraud_signal = (
            fraud_score >= 1 or consistency < 0.5 or consistency_flags >= 1
        )

        if identity_status == "FAILED":
            return (
                LoanAction.REJECT,
                max(confidence, 0.95),
                "KYCVerifier triggered a hard fraud veto because bank KYC failed.",
            )

        if identity_status == "PENDING" and extra_fraud_signal:
            return (
                LoanAction.REJECT,
                max(confidence, 0.9),
                "Pending bank KYC combined with additional fraud signals was "
                "treated as a likely fraud escalation, triggering a hard veto.",
            )

        if identity_status == "PENDING":
            return (
                LoanAction.REQUEST_MORE_INFO,
                max(confidence, 0.75),
                "Pending bank KYC requires human verification "
                "before clearing the application on fraud grounds.",
            )

        return decision, confidence, reasoning

    def _fallback(
        self,
        obs: LoanObservation,
        sub_reports: List[SubAgentReport],
        identity_status: str,
        fraud_score: int,
        consistency_flags: int,
    ) -> HeadAgentVote:
        """Deterministic fallback with identity-veto safeguards."""
        consistency = obs.application_consistency
        decision, confidence, reasoning = self._apply_identity_guards(
            decision=LoanAction.APPROVE,
            confidence=0.8,
            reasoning="Fallback: defaulting to no major fraud concern.",
            identity_status=identity_status,
            fraud_score=fraud_score,
            consistency=consistency,
            consistency_flags=consistency_flags,
        )

        if decision != LoanAction.APPROVE:
            return HeadAgentVote(
                agent_name="FraudAgent",
                decision=decision,
                reasoning=reasoning,
                confidence=confidence,
                sub_agent_reports=sub_reports,
            )

        if fraud_score >= 2 or consistency < 0.3:
            decision = LoanAction.REJECT
            confidence = 0.9
            reasoning = (
                f"Fallback: Strong fraud signal. "
                f"fraud_score={fraud_score}, consistency={consistency:.2f}."
            )
        elif fraud_score == 1 or consistency_flags >= 1:
            decision = LoanAction.REQUEST_MORE_INFO
            confidence = 0.65
            reasoning = (
                "Fallback: Moderate fraud signal or inconsistency concern. "
                f"fraud_score={fraud_score}, consistency_flags={consistency_flags}."
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
