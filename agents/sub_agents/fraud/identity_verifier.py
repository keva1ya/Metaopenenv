"""
F3: KYC Verifier Sub-Agent.

Simple bank-KYC status checker.
This agent does not infer deep identity risk. It only reads the
bank-owned KYC verification status already present on the application.
"""

from agents.sub_agents.risk.credit_analyst import SubAgentReport
from env.observation import LoanObservation


class IdentityVerifierAgent:
    """
    Compatibility wrapper for a simple KYC verification agent.

    The class name is kept stable to avoid touching imports everywhere,
    but the emitted agent name is KYCVerifier and the logic is intentionally
    lightweight.
    """

    def analyze(self, obs: LoanObservation) -> SubAgentReport:
        status = obs.bank_kyc_status

        if status == "VERIFIED":
            recommendation = "APPROVE"
            confidence = 0.95
            reasoning = (
                "Bank KYC verification is complete and verified. "
                "No additional identity review is needed from this sub-agent."
            )
            flags = ["bank_kyc_verified"]
        elif status == "FAILED":
            recommendation = "REJECT"
            confidence = 0.98
            reasoning = (
                "Bank KYC verification failed. "
                "This is a direct identity-fraud veto signal for the fraud stack."
            )
            flags = ["bank_kyc_failed"]
        else:
            recommendation = "REVIEW"
            confidence = 0.85
            reasoning = (
                "Bank KYC verification is still pending. "
                "Human verification or additional documentation is required."
            )
            flags = ["bank_kyc_pending"]

        return SubAgentReport(
            agent_name="KYCVerifier",
            recommendation=recommendation,
            confidence=confidence,
            reasoning=reasoning,
            key_flags=flags,
        )
