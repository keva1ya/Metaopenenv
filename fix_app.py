import re

with open('app.py', 'r') as f:
    content = f.read()

# Add pandas import
if 'import pandas as pd' not in content:
    content = content.replace('import numpy as np', 'import numpy as np\nimport pandas as pd\nimport io')

# Add globals
globals_str = """
dataset_df = None
dataset_index = 0
"""
if 'dataset_df = None' not in content:
    content = content.replace('current_council: Optional[VoteResult] = None\nlast_override_info: str = "No overrides yet"', 'current_council: Optional[VoteResult] = None\nlast_override_info: str = "No overrides yet"\n' + globals_str)

# Add handler functions before create_gradio_app
handlers = """
def handle_dataset_upload(file):
    global dataset_df, dataset_index
    if file is None:
        return ("No file uploaded.",) + tuple(gr.update() for _ in range(19))
    
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith('.json'):
            df = pd.read_json(file.name)
        else:
            return ("Unsupported file format.",) + tuple(gr.update() for _ in range(19))
        
        dataset_df = df
        dataset_index = 0
        return (f"Successfully loaded {len(df)} rows. Click 'Load Next Row'.",) + tuple(gr.update() for _ in range(19))
    except Exception as e:
        return (f"Error loading dataset: {e}",) + tuple(gr.update() for _ in range(19))

def load_next_from_dataset():
    global dataset_df, dataset_index
    if dataset_df is None or len(dataset_df) == 0:
        return ("No dataset loaded.",) + tuple(gr.update() for _ in range(19))
    
    if dataset_index >= len(dataset_df):
        dataset_index = 0
        
    row = dataset_df.iloc[dataset_index]
    dataset_index += 1
    
    return (
        f"Loaded row {dataset_index}/{len(dataset_df)}",
        float(row.get('income', 50000)),
        int(row.get('credit_score', 700)),
        int(row.get('employment_years', 5)),
        float(row.get('loan_amount', 100000)),
        str(row.get('loan_type', 'Personal')),
        float(row.get('mortgage_value', 0)),
        int(row.get('loan_term', 10)),
        float(row.get('existing_debt', 5000)),
        int(row.get('age', 35)),
        float(row.get('monthly_expenses', 2000)),
        int(row.get('num_dependents', 0)),
        int(row.get('previous_defaults', 0)),
        bool(row.get('gst_registered', False)),
        int(row.get('itr_filed_years', 3)),
        float(row.get('customer_score', 0.8)),
        float(row.get('spending_volatility', 0.5)),
        float(row.get('loan_purpose_clarity', 0.8)),
        float(row.get('application_consistency', 0.9)),
        str(row.get('bank_kyc_status', 'VERIFIED'))
    )

"""
if 'def handle_dataset_upload' not in content:
    content = content.replace('def create_gradio_app() -> gr.Blocks:', handlers + 'def create_gradio_app() -> gr.Blocks:')

# Replace process_next_applicant
process_func_new = """def process_next_applicant(
    # 19 fields
    income, credit_score, employment_years, loan_amount, loan_type, mortgage_value,
    loan_term, existing_debt, age, monthly_expenses, num_dependents, previous_defaults,
    gst_registered, itr_filed_years, customer_score, spending_volatility,
    loan_purpose_clarity, application_consistency, bank_kyc_status,
    # action & weights
    selected_action: str, 
    risk_weight: int,
    fraud_weight: int,
    policy_weight: int,
):
    global current_council, last_override_info

    if (risk_weight + fraud_weight + policy_weight) != 100:
        raise gr.Error("Scoring multipliers must sum to exactly 100.")

    try:
        obs = LoanObservation(
            income=float(income),
            credit_score=int(credit_score),
            employment_years=int(employment_years),
            loan_amount=float(loan_amount),
            loan_type=str(loan_type),
            mortgage_value=float(mortgage_value),
            loan_term=int(loan_term),
            existing_debt=float(existing_debt),
            age=int(age),
            monthly_expenses=float(monthly_expenses),
            num_dependents=int(num_dependents),
            previous_defaults=int(previous_defaults),
            gst_registered=bool(gst_registered),
            itr_filed_years=int(itr_filed_years),
            customer_score=float(customer_score),
            spending_volatility=float(spending_volatility),
            loan_purpose_clarity=float(loan_purpose_clarity),
            application_consistency=float(application_consistency),
            bank_kyc_status=str(bank_kyc_status)
        )
    except Exception as e:
        raise gr.Error(f"Invalid input data: {e}")

    env.set_observation(obs)
    env.episode += 1

    # Check mortgage mode"""

# Regex to replace process_next_applicant up to "# Check mortgage mode"
content = re.sub(r'def process_next_applicant\(.*?# Check mortgage mode', process_func_new, content, flags=re.DOTALL)

# Now fix the wiring
wiring_fix = """
        input_fields = [
            inp_income, inp_credit_score, inp_employment_years, inp_loan_amount, inp_loan_type,
            inp_mortgage_value, inp_loan_term, inp_existing_debt, inp_age, inp_monthly_expenses,
            inp_num_dependents, inp_previous_defaults, inp_gst_registered, inp_itr_filed_years,
            inp_customer_score, inp_spending_volatility, inp_loan_purpose_clarity,
            inp_application_consistency, inp_bank_kyc_status
        ]

        dataset_file.upload(
            fn=handle_dataset_upload,
            inputs=[dataset_file],
            outputs=[upload_status] + input_fields
        )

        load_next_btn.click(
            fn=load_next_from_dataset,
            inputs=[],
            outputs=[upload_status] + input_fields
        )

        next_flow = next_btn.click(
"""

if 'dataset_file.upload' not in content:
    content = content.replace('next_flow = next_btn.click(', wiring_fix)

next_flow_inputs_fix = """        next_flow.then(
            fn=process_next_applicant,
            inputs=[
                inp_income, inp_credit_score, inp_employment_years, inp_loan_amount, inp_loan_type,
                inp_mortgage_value, inp_loan_term, inp_existing_debt, inp_age, inp_monthly_expenses,
                inp_num_dependents, inp_previous_defaults, inp_gst_registered, inp_itr_filed_years,
                inp_customer_score, inp_spending_volatility, inp_loan_purpose_clarity,
                inp_application_consistency, inp_bank_kyc_status,
                action_radio, 
                risk_weight_slider,
                fraud_weight_slider,
                policy_weight_slider
            ],"""

content = re.sub(r'next_flow\.then\(\s*fn=process_next_applicant,\s*inputs=\[.*?\]\,', next_flow_inputs_fix, content, flags=re.DOTALL)

with open('app.py', 'w') as f:
    f.write(content)

