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
    income: float = Field(ge=30000, le=200000, description="Annual income in currency units")
    credit_score: int = Field(ge=300, le=850, description="Credit bureau score")
    employment_years: int = Field(ge=0, le=40, description="Years of continuous employment")
    loan_amount: float = Field(ge=50000, le=500000, description="Requested loan amount")
    mortgage_value: float = Field(ge=0, le=700000, description="Existing mortgage/collateral value")
    loan_term: int = Field(ge=5, le=30, description="Loan term in years")
    existing_debt: float = Field(ge=0, le=200000, description="Total existing debt obligations")
    age: int = Field(ge=18, le=65, description="Applicant age")
    monthly_expenses: float = Field(ge=5000, le=80000, description="Monthly recurring expenses")
    num_dependents: int = Field(ge=0, le=8, description="Number of financial dependents")
    previous_defaults: int = Field(ge=0, le=5, description="Count of previous loan defaults")
    gst_registered: bool = Field(description="Whether the applicant is GST registered")
    itr_filed_years: int = Field(ge=0, le=20, description="Years of income tax returns filed")
    customer_score: float = Field(ge=0.0, le=1.0, description="Internal loyalty/customer score")

    # ── Behavioral Fields ────────────────────────────────────
    spending_volatility: float = Field(ge=0.0, le=1.0, description="Variance in spending patterns")
    loan_purpose_clarity: float = Field(ge=0.0, le=1.0, description="Clarity of stated loan purpose")
    application_consistency: float = Field(ge=0.0, le=1.0, description="Consistency across application fields")
    bank_kyc_status: Literal["VERIFIED", "PENDING", "FAILED"] = Field(
        description="Whether bank KYC verification is verified, pending, or failed"
    )
