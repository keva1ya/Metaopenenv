# Implementation Plan: Replacing Synthetic Data with Real Datasets & Manual Entry

## Objective
Completely remove the `SyntheticDataGenerator` from the `OpenEnvR2/Metaopenenv` project and introduce user-provided data capabilities. This involves adding manual entry fields for all loan parameters in the Gradio UI and supporting CSV dataset uploads for batch/RL processing.

## Step-by-Step Execution Plan

### Step 1: Remove Synthetic Data Generation
1. Delete `env/data_generator.py`.
2. Update `env/loan_env.py`:
   - Remove all imports and initializations of `SyntheticDataGenerator`.
   - Modify `reset()` so it no longer auto-generates data. Instead, it will prepare the environment for a new observation (e.g., just incrementing the episode counter) or pop one from a loaded dataset queue.
   - Add a method `set_observation(obs: LoanObservation)` to allow injecting manual or dataset-driven data into the environment.
3. Update `env/__init__.py` and other imports referencing `data_generator.py`.

### Step 2: Update the Gradio UI (`app.py`) for Manual Entry
1. In the UI layout, add a new section (e.g., an Accordion or Tab) for "Applicant Data Entry".
2. Create Gradio input components for all 18 parameters:
   - **Financials**: Income, Credit Score, Employment Years, Loan Amount, Loan Type, Mortgage Value, Loan Term, Existing Debt, Age, Monthly Expenses, Dependents, Previous Defaults, GST Registered, ITR Filed Years, Customer Score.
   - **Behaviorals**: Spending Volatility, Loan Purpose Clarity, Application Consistency, Bank KYC Status.
3. Wire these inputs so that a new "Process Application" button reads from these fields, constructs a `LoanObservation`, and passes it to the `LoanEnv`.

### Step 3: Implement Dataset Upload in Gradio (`app.py`)
1. Add a `gr.File` component to allow users to upload a CSV dataset.
2. Create a backend function to parse the uploaded CSV using `pandas` and store the records in a global state or directly in `LoanEnv`.
3. Add a "Load Next from Dataset" button that pops the next row from the uploaded dataset, auto-fills the manual entry fields, and optionally processes the application.

### Step 4: Adjust Backend and API
1. Update `app.py`'s FastAPI `/reset` and `/step` endpoints to accommodate external data injection. For `/reset`, it might need to accept an observation payload instead of generating one, or a new `/inject_observation` endpoint can be created.
2. Update offline scripts (e.g., `scripts/test_offline.py`) to load a sample dataset instead of using synthetic data.

### Step 5: Final Review & Testing
1. Review the UI components to ensure they match the aesthetic of the current app.
2. Verify that the RL trace buffer and GRPO trainer still function correctly with the manually provided or dataset-provided observations.
3. Remove any lingering code referencing `SyntheticDataGenerator`.

## Expected Outcome
The system will no longer "hallucinate" or randomly generate loan applications. Every decision made by the sub-agents and the weighted engine will be based entirely on ground-truth data provided either via manual UI entry or a structured CSV dataset.
