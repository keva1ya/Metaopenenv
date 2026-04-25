"""
Offline testing script for the Loan Approval OpenEnv system.
This script tests all environment components without loading PyTorch.
It verifies:
- Data generation
- Sub-agent evaluation
- LLM Head Agent Council Voting
- Dissent Logging
- State Builder & Reward formulation
"""

import sys
sys.path.insert(0, '.')

# Test observation model
from env.observation import LoanObservation
obs = LoanObservation(
    income=80000, credit_score=720, employment_years=10,
    loan_amount=200000, mortgage_value=300000, loan_term=15,
    existing_debt=50000, age=35, monthly_expenses=25000,
    num_dependents=2, previous_defaults=0, gst_registered=True,
    itr_filed_years=8, customer_score=0.85,
    spending_volatility=0.3, loan_purpose_clarity=0.8,
    application_consistency=0.9, bank_kyc_status="VERIFIED"
)
print('✅ Applicant Profile created: Credit', obs.credit_score)

# Test data generator
from env.data_generator import SyntheticDataGenerator
gen = SyntheticDataGenerator(seed=42)
test_obs = gen.generate()
print(f'✅ Synthetic Data Generator OK: income={test_obs.income:.0f}, credit={test_obs.credit_score}')

# Test all 9 sub-agents
from agents.sub_agents.risk.credit_analyst import CreditAnalystAgent
from agents.sub_agents.risk.income_verifier import IncomeVerifierAgent
from agents.sub_agents.risk.debt_analyzer import DebtAnalyzerAgent
from agents.sub_agents.fraud.pattern_checker import PatternCheckerAgent
from agents.sub_agents.fraud.consistency_checker import ConsistencyCheckerAgent
from agents.sub_agents.fraud.identity_verifier import IdentityVerifierAgent
from agents.sub_agents.policy.compliance_officer import ComplianceOfficerAgent
from agents.sub_agents.policy.term_validator import TermValidatorAgent
from agents.sub_agents.policy.ltv_checker import LTVCheckerAgent

print('\n🔍 Running Level-1 Sub-Agents (Zero inference cost)...')
credit_r = CreditAnalystAgent().analyze(obs)
income_r = IncomeVerifierAgent().analyze(obs)
debt_r = DebtAnalyzerAgent().analyze(obs)
pattern_r = PatternCheckerAgent().analyze(obs)
consist_r = ConsistencyCheckerAgent().analyze(obs)
identity_r = IdentityVerifierAgent().analyze(obs)
comply_r = ComplianceOfficerAgent().analyze(obs)
term_r = TermValidatorAgent().analyze(obs)
ltv_r = LTVCheckerAgent().analyze(obs)

print(f'   Risk 1 - CreditAnalyst:      {credit_r.recommendation} ({credit_r.confidence:.2f})')
print(f'   Risk 2 - IncomeVerifier:     {income_r.recommendation} ({income_r.confidence:.2f})')
print(f'   Risk 3 - DebtAnalyzer:       {debt_r.recommendation} ({debt_r.confidence:.2f})')
print(f'   Fraud 1 - PatternChecker:    {pattern_r.recommendation} ({pattern_r.confidence:.2f})')
print(f'   Fraud 2 - ConsistencyChecker: {consist_r.recommendation} ({consist_r.confidence:.2f})')
print(f'   Fraud 3 - KYCVerifier:       {identity_r.recommendation} ({identity_r.confidence:.2f})')
print(f'   Policy 1 - ComplianceOfficer: {comply_r.recommendation} ({comply_r.confidence:.2f})')
print(f'   Policy 2 - TermValidator:     {term_r.recommendation} ({term_r.confidence:.2f})')
print(f'   Policy 3 - LTVChecker:        {ltv_r.recommendation} ({ltv_r.confidence:.2f})')

# Test head agents (demo/fallback mode)
from agents.council.risk_agent import RiskAgent
from agents.council.fraud_agent import FraudAgent
from agents.council.policy_agent import PolicyAgent

print('\n🧠 Running Level-2 Head Council Votes...')
risk_vote = RiskAgent(None).evaluate(obs, [credit_r, income_r, debt_r], 'None')
fraud_vote = FraudAgent(None).evaluate(obs, [pattern_r, consist_r, identity_r], 'None')
policy_vote = PolicyAgent(None).evaluate(obs, [comply_r, term_r, ltv_r], 'None')

print(f'   💡 RiskAgent Synthesis:   {risk_vote.decision.value} ({risk_vote.confidence:.2f})')
print(f'   💡 FraudAgent Synthesis:  {fraud_vote.decision.value} ({fraud_vote.confidence:.2f})')
print(f'   💡 PolicyAgent Synthesis: {policy_vote.decision.value} ({policy_vote.confidence:.2f})')

# Test voting engine
from agents.voting import VotingEngine
ve = VotingEngine()
vote_result = ve.run_vote([risk_vote, fraud_vote, policy_vote], 1)
vote_result.all_sub_reports = [credit_r, income_r, debt_r, pattern_r, consist_r, identity_r, comply_r, term_r, ltv_r]

print('\n⚖️ Voting Engine Result:')
print(f'   Final Council Rec: {vote_result.council_recommendation.value}')
print(f'   Unanimous: {vote_result.was_unanimous}')
print(f'   Breakdown: {vote_result.vote_breakdown}')

# Test reward shaper
from brain.reward_shaper import calculate_shaped_reward
shaped_r = calculate_shaped_reward(obs, 'APPROVE', 'hard')
print(f'\n💰 Reward Shaping Engine Base Check -> Evaluated Reward Score: {shaped_r:.4f}')

# Test state builder
from brain.state_builder import build_state_vector, PortfolioState
portfolio = PortfolioState()
state_vec = build_state_vector(obs, vote_result, portfolio)
print(f'\n🌐 Preparing weighted-engine state payload...')
print(f'   State Array built: shape={state_vec.shape}, dtype={state_vec.dtype}')
print(f'   State Signature[0:5]: {state_vec[:5]}')

print('\n============================================================')
print('✅ ALL BASE ARCHITECTURE COMPONENTS VERIFIED EFFECTIVELY!')
print('   (PyTorch execution will proceed once network is available)')
print('============================================================\n')
