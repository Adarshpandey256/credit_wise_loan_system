# Credit Wise — Loan Approval Prediction System

A Streamlit app that predicts loan approval using a Logistic Regression
model trained on applicant financial and demographic data, deployed via
Streamlit Community Cloud.

## ⚠️ Read this before you deploy

1. **The CSV you uploaded alongside the notebook does not match the
   notebook's training data.** The notebook was trained on
   `loan_approval_data.csv` (20 columns: `Applicant_Income`,
   `Credit_Score`, `Employment_Status`, `Loan_Purpose`, etc., ~1000 rows).
   The file you gave me (`Loan_Approval_Prediction.csv`, 7 columns,
   20 rows) is a different schema entirely and can't be used to train
   this model. **You need to supply the real `loan_approval_data.csv`**
   before running `train_model.py`.
2. **Two dropdown category lists in `app.py` are incomplete.** During
   training, `OneHotEncoder(drop="first")` drops one category per column
   as a baseline. For `Employment_Status` and `Employer_Category`, that
   dropped category never appeared anywhere in the notebook's visible
   output, so it couldn't be reconstructed. Once you have the real CSV,
   run:
   ```python
   df["Employment_Status"].unique()
   df["Employer_Category"].unique()
   ```
   and add any missing value to the `st.selectbox(...)` option lists in
   `app.py` (marked with `# TODO` comments).
3. **A bug was fixed, not reproduced:** the notebook's imputation step
   accidentally applied `most_frequent` imputation to the target column
   `Loan_Approved` itself, silently fabricating labels for rows with a
   missing outcome. `train_model.py` instead drops rows with a missing
   target.

## Project files

```
credit-wise-loan/
├── app.py                 # Streamlit app (UI + prediction)
├── train_model.py         # Trains the model and saves model_pipeline.pkl
├── pipeline_utils.py       # Shared preprocessing logic (imported by both files)
├── model_pipeline.pkl      # Trained model (created by train_model.py — not in git until you run it)
├── requirements.txt
├── loan_approval_data.csv  # Your real training data (you must add this)
└── README.md
```

`pipeline_utils.py` must exist next to both `train_model.py` and `app.py` —
the saved pipeline is a Python object that references the custom
`FeatureEngineer` class defined there, and joblib needs that class
importable to unpickle the file.

## What the model does

- **Target:** `Loan_Approved` (Yes/No)
- **Model:** Logistic Regression
- **Inputs:** 11 numeric fields (income, age, credit score, loan amount,
  etc.) and 7 categorical fields (employment status, marital status,
  loan purpose, property area, education level, gender, employer
  category)
- **Feature engineering:** adds `DTI_Ratio_sq`, `Credit_Score_sq`, and
  `Applicant_Income_log`, and drops the raw `Credit_Score`/`DTI_Ratio`
  columns in favor of the squared versions — this matches the notebook's
  final, best-performing feature set exactly.
- Everything (imputation, feature engineering, one-hot encoding,
  scaling, and the classifier) is bundled into **one scikit-learn
  `Pipeline`** saved as `model_pipeline.pkl`, so `app.py` never
  duplicates preprocessing logic.

## Running locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your real `loan_approval_data.csv` in this folder.
3. Train the model (creates `model_pipeline.pkl`):
   ```bash
   python train_model.py
   ```
4. Launch the app:
   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Community Cloud

See the step-by-step deployment guide provided alongside this README.
