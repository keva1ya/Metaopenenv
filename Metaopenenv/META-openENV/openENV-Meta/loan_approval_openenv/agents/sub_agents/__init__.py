"""Sub-agents package: rule-based analyst agents."""

from agents.sub_agents.risk.credit_analyst import CreditAnalystAgent
from agents.sub_agents.risk.income_verifier import IncomeVerifierAgent
from agents.sub_agents.risk.debt_analyzer import DebtAnalyzerAgent
from agents.sub_agents.fraud.pattern_checker import PatternCheckerAgent
from agents.sub_agents.fraud.consistency_checker import ConsistencyCheckerAgent
from agents.sub_agents.policy.compliance_officer import ComplianceOfficerAgent
from agents.sub_agents.policy.term_validator import TermValidatorAgent
from agents.sub_agents.policy.ltv_checker import LTVCheckerAgent

__all__ = [
    "CreditAnalystAgent",
    "IncomeVerifierAgent",
    "DebtAnalyzerAgent",
    "PatternCheckerAgent",
    "ConsistencyCheckerAgent",
    "ComplianceOfficerAgent",
    "TermValidatorAgent",
    "LTVCheckerAgent",
]
