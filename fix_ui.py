import re

with open('app.py', 'r') as f:
    content = f.read()

components = """
        # ── MANUAL ENTRY / DATASET INPUT ────────────────────
        gr.Markdown("---")
        gr.Markdown("### 📝 Applicant Data Entry / Dataset Upload")
        with gr.Accordion("Upload Dataset (CSV/JSON)", open=False):
            with gr.Row():
                dataset_file = gr.File(label="Upload Dataset File", file_types=[".csv", ".json"])
            with gr.Row():
                upload_status = gr.Textbox(label="Dataset Status", interactive=False)
            with gr.Row():
                load_next_btn = gr.Button("⬇️ Load Next Row from Dataset")

        with gr.Accordion("Manual Data Entry", open=True):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Financial Data")
                    inp_income = gr.Number(label="Income", value=80000)
                    inp_credit_score = gr.Number(label="Credit Score", value=750)
                    inp_employment_years = gr.Number(label="Employment Years", value=5)
                    inp_loan_amount = gr.Number(label="Loan Amount", value=200000)
                    inp_loan_type = gr.Dropdown(choices=["Personal", "Business", "Mortgage", "Auto", "Student", "Home/Land"], value="Personal", label="Loan Type")
                    inp_mortgage_value = gr.Number(label="Mortgage Value", value=0)
                    inp_loan_term = gr.Number(label="Loan Term", value=15)
                    inp_existing_debt = gr.Number(label="Existing Debt", value=10000)
                    inp_age = gr.Number(label="Age", value=35)
                    inp_monthly_expenses = gr.Number(label="Monthly Expenses", value=3000)
                with gr.Column():
                    gr.Markdown("#### Additional Info & Behavioral Data")
                    inp_num_dependents = gr.Number(label="Dependents", value=2)
                    inp_previous_defaults = gr.Number(label="Previous Defaults", value=0)
                    inp_gst_registered = gr.Checkbox(label="GST Registered", value=False)
                    inp_itr_filed_years = gr.Number(label="ITR Filed Years", value=4)
                    inp_customer_score = gr.Slider(minimum=0, maximum=1, value=0.85, label="Customer Score")
                    inp_spending_volatility = gr.Slider(minimum=0, maximum=1, value=0.4, label="Spending Volatility")
                    inp_loan_purpose_clarity = gr.Slider(minimum=0, maximum=1, value=0.9, label="Loan Purpose Clarity")
                    inp_application_consistency = gr.Slider(minimum=0, maximum=1, value=0.95, label="Application Consistency")
                    inp_bank_kyc_status = gr.Dropdown(choices=["VERIFIED", "PENDING", "FAILED"], value="VERIFIED", label="Bank KYC Status")

"""

# Insert these components right after the header section in create_gradio_app
insert_regex = r"(</section>\s*\"\"\"\s*\)\s*)"

if 'inp_income = gr.Number' not in content:
    match = re.search(insert_regex, content)
    if match:
        insert_point = match.end()
        content = content[:insert_point] + "\n" + components + content[insert_point:]
        with open('app.py', 'w') as f:
            f.write(content)
        print("UI components added.")
    else:
        print("Could not find insert point.")
else:
    print("UI components already exist.")
