"""
Synthetic data generator for loan applicant observations.

Uses numpy's new-style Generator (default_rng) exclusively.
Implements realistic correlations between fields.
Never uses random or numpy.random module-level functions.
"""

import numpy as np

from env.observation import LoanObservation


class SyntheticDataGenerator:
    """
    Generates synthetic loan applicant data with realistic
    correlations between financial and behavioral fields.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(self) -> LoanObservation:
        """
        Generate one complete LoanObservation with all 17 fields.

        Correlations implemented:
          1. spending_volatility increases for low-income + high-loan
          2. application_consistency decreases with high defaults
          3. customer_score correlated with defaults & employment
          4. gst_registered more likely for higher incomes
        """
        # ── Core financial fields (uniform random in range) ──
        income = float(self.rng.uniform(30000, 200000))
        credit_score = int(self.rng.integers(300, 851))
        employment_years = int(self.rng.integers(0, 41))
        loan_amount = float(self.rng.uniform(50000, 500000))
        mortgage_value = float(self.rng.uniform(0, 700000))
        loan_term = int(self.rng.integers(5, 31))
        existing_debt = float(self.rng.uniform(0, 200000))
        age = int(self.rng.integers(18, 66))
        monthly_expenses = float(self.rng.uniform(5000, 80000))
        num_dependents = int(self.rng.integers(0, 9))
        previous_defaults = int(self.rng.integers(0, 6))
        itr_filed_years = int(self.rng.integers(0, 21))

        # ── Correlation 4: gst_registered ────────────────────
        if income > 80000:
            gst_registered = bool(self.rng.random() < 0.4)
        else:
            gst_registered = False

        # ── Correlation 3: customer_score ────────────────────
        if previous_defaults == 0 and employment_years > 5:
            customer_score = float(self.rng.uniform(0.6, 1.0))
        elif previous_defaults > 2:
            customer_score = float(self.rng.uniform(0.0, 0.4))
        else:
            customer_score = float(self.rng.uniform(0.0, 1.0))

        # ── Correlation 1: spending_volatility ───────────────
        spending_volatility_base = float(self.rng.uniform(0, 1))
        if income < 50000 and loan_amount > 200000:
            spending_volatility_base = min(1.0, spending_volatility_base + 0.3)
        spending_volatility = spending_volatility_base

        # ── Behavioral: loan_purpose_clarity (uniform) ───────
        loan_purpose_clarity = float(self.rng.uniform(0, 1))

        # ── Correlation 2: application_consistency ───────────
        application_consistency_base = float(self.rng.uniform(0, 1))
        defaults_over_two = max(0, previous_defaults - 2)
        for _ in range(defaults_over_two):
            application_consistency_base = max(0.0, application_consistency_base - 0.2)
        application_consistency = application_consistency_base

        return LoanObservation(
            income=income,
            credit_score=credit_score,
            employment_years=employment_years,
            loan_amount=loan_amount,
            mortgage_value=mortgage_value,
            loan_term=loan_term,
            existing_debt=existing_debt,
            age=age,
            monthly_expenses=monthly_expenses,
            num_dependents=num_dependents,
            previous_defaults=previous_defaults,
            gst_registered=gst_registered,
            itr_filed_years=itr_filed_years,
            customer_score=customer_score,
            spending_volatility=spending_volatility,
            loan_purpose_clarity=loan_purpose_clarity,
            application_consistency=application_consistency,
        )
