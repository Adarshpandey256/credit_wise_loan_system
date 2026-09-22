"""
app.py
------
Streamlit app for the Credit Wise Loan Prediction System.

Loads the pre-trained pipeline (model_pipeline.pkl) produced by
train_model.py and does NOT retrain anything at runtime.
"""

import joblib
import pandas as pd
import streamlit as st

# Required so joblib can unpickle the FeatureEngineer class stored inside
# model_pipeline.pkl. This import looks unused but must stay.
from pipeline_utils import FeatureEngineer  # noqa: F401

MODEL_PATH = "model_pipeline.pkl"

st.set_page_config(
    page_title="Credit Wise — Loan Approval Predictor",
    page_icon="💳",
    layout="centered",
)


@st.cache_resource
def load_pipeline():
    """Load the trained pipeline once and cache it across reruns/sessions."""
    return joblib.load(MODEL_PATH)


def get_pipeline():
    try:
        return load_pipeline(), None
    except FileNotFoundError:
        return None, (
            f"Couldn't find '{MODEL_PATH}'. Run train_model.py first and make "
            "sure model_pipeline.pkl is in the same folder as app.py."
        )
    except Exception as e:  # noqa: BLE001
        return None, f"Failed to load the model: {e}"


st.title("💳 Credit Wise — Loan Approval Predictor")
st.write(
    "Fill in the applicant's details below and click **Predict** to see "
    "whether the loan is likely to be approved."
)

pipeline, load_error = get_pipeline()
if load_error:
    st.error(load_error)
    st.stop()

# -----------------------------------------------------------------------
# NOTE on the dropdown options below:
# All values EXCEPT Employment_Status and Employer_Category were confirmed
# directly from the notebook's data. For those two columns, one category
# each was dropped by OneHotEncoder during training and never appeared in
# the notebook's visible output, so it could not be reconstructed reliably.
# Before deploying, run this on your real training CSV and update the two
# lists marked below if anything is missing:
#   df["Employment_Status"].unique()
#   df["Employer_Category"].unique()
# -----------------------------------------------------------------------

with st.form("loan_form"):
    st.subheader("Applicant details")

    col1, col2 = st.columns(2)

    with col1:
        applicant_income = st.number_input(
            "Applicant Income (monthly)", min_value=0.0, value=15000.0, step=500.0
        )
        coapplicant_income = st.number_input(
            "Coapplicant Income (monthly)", min_value=0.0, value=0.0, step=500.0
        )
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        dependents = st.number_input(
            "Number of Dependents", min_value=0, max_value=10, value=0, step=1
        )
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=900, value=650, step=1
        )
        existing_loans = st.number_input(
            "Existing Loans", min_value=0, max_value=10, value=0, step=1
        )
        dti_ratio = st.slider("DTI Ratio (debt-to-income)", 0.0, 1.0, 0.30, 0.01)

    with col2:
        savings = st.number_input(
            "Savings", min_value=0.0, value=10000.0, step=500.0
        )
        collateral_value = st.number_input(
            "Collateral Value", min_value=0.0, value=20000.0, step=500.0
        )
        loan_amount = st.number_input(
            "Loan Amount Requested", min_value=0.0, value=20000.0, step=500.0
        )
        loan_term = st.number_input(
            "Loan Term (months)", min_value=1, max_value=480, value=60, step=1
        )
        employment_status = st.selectbox(
            "Employment Status",
            # TODO: add the missing 4th category once confirmed from your real data
            ["Salaried", "Self-employed", "Unemployed"],
        )
        marital_status = st.selectbox("Marital Status", ["Married", "Single"])
        gender = st.selectbox("Gender", ["Female", "Male"])

    st.subheader("Loan & background details")
    col3, col4 = st.columns(2)
    with col3:
        loan_purpose = st.selectbox(
            "Loan Purpose", ["Business", "Car", "Education", "Home", "Personal"]
        )
        property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
    with col4:
        education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        employer_category = st.selectbox(
            "Employer Category",
            # TODO: add the missing 5th category once confirmed from your real data
            ["Government", "MNC", "Private", "Unemployed"],
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_row = pd.DataFrame(
        [
            {
                "Applicant_Income": applicant_income,
                "Coapplicant_Income": coapplicant_income,
                "Age": age,
                "Dependents": dependents,
                "Credit_Score": credit_score,
                "Existing_Loans": existing_loans,
                "DTI_Ratio": dti_ratio,
                "Savings": savings,
                "Collateral_Value": collateral_value,
                "Loan_Amount": loan_amount,
                "Loan_Term": loan_term,
                "Employment_Status": employment_status,
                "Marital_Status": marital_status,
                "Loan_Purpose": loan_purpose,
                "Property_Area": property_area,
                "Education_Level": education_level,
                "Gender": gender,
                "Employer_Category": employer_category,
            }
        ]
    )

    try:
        prediction = pipeline.predict(input_row)[0]
        proba = None
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(input_row)[0][1]  # P(approved)
    except Exception as e:  # noqa: BLE001
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.divider()
    if prediction == 1:
        st.success("✅ Loan likely **Approved**")
    else:
        st.error("❌ Loan likely **Not Approved**")

    if proba is not None:
        st.metric("Approval probability", f"{proba * 100:.1f}%")

    with st.expander("See the inputs sent to the model"):
        st.dataframe(input_row)
