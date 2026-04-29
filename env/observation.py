"""
Observation model for the Loan Approval Environment.

Uses Pydantic v2 with frozen immutability.
All fields are validated with strict ranges.

RULE: Financial fields → RiskAgent and PolicyAgent only.
RULE: Behavioral fields → FraudAgent only.
This separation is enforced by prompt content in head agents.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LoanObservation(BaseModel):
    """
    Immutable observation representing a single loan applicant.

    Financial Fields (14):
        Used by RiskAgent and PolicyAgent sub-agents/heads.
    Behavioral and KYC Fields (4):
        Used by FraudAgent sub-agents/heads only.
    """

    model_config = ConfigDict(frozen=True)

    # ── Financial Fields ─────────────────────────────────────
    income: float = Field(description="Annual income in currency units")
    credit_score: int = Field(description="Credit bureau score")
    employment_years: int = Field(description="Years of continuous employment")
    loan_amount: float = Field(description="Requested loan amount")
    loan_type: Literal["Personal", "Business", "Mortgage", "Auto", "Student", "Home/Land"] = Field(
        description="The category or purpose of the requested loan"
    )
    mortgage_value: float = Field(description="Existing mortgage/collateral value")
    loan_term: int = Field(description="Loan term in years")
    existing_debt: float = Field(description="Total existing debt obligations")
    age: int = Field(description="Applicant age")
    monthly_expenses: float = Field(description="Monthly recurring expenses")
    num_dependents: int = Field(description="Number of financial dependents")
    previous_defaults: int = Field(description="Count of previous loan defaults")
    gst_registered: bool = Field(description="Whether the applicant is GST registered")
    itr_filed_years: int = Field(description="Years of income tax returns filed")
    customer_score: float = Field(description="Internal loyalty/customer score")

    # ── Behavioral Fields ────────────────────────────────────
    spending_volatility: float = Field(description="Variance in spending patterns")
    loan_purpose_clarity: float = Field(description="Clarity of stated loan purpose")
    application_consistency: float = Field(description="Consistency across application fields")
    bank_kyc_status: Literal["VERIFIED", "PENDING", "FAILED"] = Field(
        description="Whether bank KYC verification is verified, pending, or failed"
    )
