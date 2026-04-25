"""
Action space for the Loan Approval Environment.

Three valid actions:
  APPROVE            — Grant the loan.
  REJECT             — Deny the loan.
  REQUEST_MORE_INFO  — Request additional documentation.
"""

from enum import Enum


class LoanAction(str, Enum):
    """Enumeration of all valid loan decisions."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REQUEST_MORE_INFO = "REQUEST_MORE_INFO"
