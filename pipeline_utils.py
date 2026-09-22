"""
pipeline_utils.py
------------------
Shared preprocessing logic for the Credit Wise Loan Prediction System.

This module must be imported by BOTH train_model.py (to build/fit the
pipeline) and app.py (to unpickle it) — a joblib/pickle file that contains
a custom class can only be loaded if that exact class is importable in the
process doing the loading.

Column lists and transform logic below are taken directly from the
uploaded notebook (credit_wise_loan_system.ipynb):
  - Numeric columns come from `numerical_col` (cell computing
    df.select_dtypes(include=['float64'])), minus Applicant_ID (dropped)
    and the target.
  - Categorical columns come from `categorical_col`, minus the target.
  - The engineered features (DTI_Ratio_sq, Credit_Score_sq,
    Applicant_Income_log) and the dropping of raw Credit_Score/DTI_Ratio
    reproduce the notebook's final feature-engineering cell exactly.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Raw numeric columns used as model inputs (Applicant_ID and the target
# Loan_Approved are excluded)
NUMERIC_COLS = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Dependents",
    "Credit_Score",
    "Existing_Loans",
    "DTI_Ratio",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
]

# Raw categorical columns used as model inputs
CATEGORICAL_COLS = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Education_Level",
    "Gender",
    "Employer_Category",
]

ALL_INPUT_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Numeric columns AFTER feature engineering (Credit_Score and DTI_Ratio
# are dropped in favor of their squared versions; Applicant_Income_log is
# added alongside the original Applicant_Income, exactly as in the
# notebook's final feature-engineering cell)
ENGINEERED_NUMERIC_COLS = [
    "Applicant_Income",
    "Coapplicant_Income",
    "Age",
    "Dependents",
    "Existing_Loans",
    "Savings",
    "Collateral_Value",
    "Loan_Amount",
    "Loan_Term",
    "DTI_Ratio_sq",
    "Credit_Score_sq",
    "Applicant_Income_log",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Reproduces the notebook's preprocessing + feature-engineering steps:

      1. Mean-imputes missing numeric values (fit on training data only).
      2. Most-frequent-imputes missing categorical values (fit on
         training data only).
      3. Adds DTI_Ratio_sq, Credit_Score_sq, Applicant_Income_log.
      4. Drops the raw Credit_Score and DTI_Ratio columns (the notebook's
         final model uses the squared versions instead of the raw ones).

    Kept as a scikit-learn compatible transformer so the whole pipeline
    (this + encoding + scaling + model) can be saved as ONE file.
    """

    def fit(self, X, y=None):
        X = X.copy()
        self.numeric_means_ = X[NUMERIC_COLS].mean()
        self.categorical_modes_ = X[CATEGORICAL_COLS].mode().iloc[0]
        return self

    def transform(self, X):
        X = X.copy()
        X[NUMERIC_COLS] = X[NUMERIC_COLS].fillna(self.numeric_means_)
        X[CATEGORICAL_COLS] = X[CATEGORICAL_COLS].fillna(self.categorical_modes_)

        X["DTI_Ratio_sq"] = X["DTI_Ratio"] ** 2
        X["Credit_Score_sq"] = X["Credit_Score"] ** 2
        X["Applicant_Income_log"] = np.log1p(X["Applicant_Income"])

        X = X.drop(columns=["Credit_Score", "DTI_Ratio"])
        return X
