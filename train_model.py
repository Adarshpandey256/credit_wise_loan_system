"""
train_model.py
---------------
Run this ONCE (locally or in Colab/Jupyter) to train the model and produce
model_pipeline.pkl, which app.py loads for predictions.

IMPORTANT: point CSV_PATH at your REAL training data — the same file the
notebook used (loan_approval_data.csv, ~1000 rows, 20 columns). The CSV you
uploaded alongside the notebook does NOT have this schema and cannot be
used to train this model. See README.md for details.

Usage:
    python train_model.py
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline_utils import (
    ALL_INPUT_COLS,
    CATEGORICAL_COLS,
    ENGINEERED_NUMERIC_COLS,
    FeatureEngineer,
)

# ---------------------------------------------------------------------
# 1. Point this at your ACTUAL training CSV (loan_approval_data.csv)
# ---------------------------------------------------------------------
CSV_PATH = "loan_approval_data.csv"
MODEL_OUT_PATH = "model_pipeline.pkl"

# ---------------------------------------------------------------------
# 2. Load and do the minimal cleanup the notebook does before modeling
# ---------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

if "Applicant_ID" in df.columns:
    df = df.drop(columns=["Applicant_ID"])

missing_cols = [c for c in ALL_INPUT_COLS + ["Loan_Approved"] if c not in df.columns]
if missing_cols:
    raise ValueError(
        f"CSV is missing expected columns: {missing_cols}. "
        "Make sure CSV_PATH points to the same dataset the notebook was trained on."
    )

# Fix: the notebook accidentally imputed the TARGET column with
# most-frequent, which silently fabricates labels for missing rows.
# We drop rows with a missing target instead.
df = df.dropna(subset=["Loan_Approved"])

X = df[ALL_INPUT_COLS].copy()
y = (df["Loan_Approved"].astype(str).str.strip().str.lower() == "yes").astype(int)

# ---------------------------------------------------------------------
# 3. Train/test split (same 80/20 split and random_state as the notebook)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------
# 4. Build the full pipeline: feature engineering -> encoding -> scaling
#    -> model, saved as a SINGLE object so app.py never has to duplicate
#    any preprocessing logic.
# ---------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", ENGINEERED_NUMERIC_COLS),
        (
            "cat",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            CATEGORICAL_COLS,
        ),
    ]
)

pipeline = Pipeline(
    steps=[
        ("feature_engineering", FeatureEngineer()),
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# ---------------------------------------------------------------------
# 5. Train and evaluate (same metrics the notebook printed)
# ---------------------------------------------------------------------
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 score:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------------------------------------------------
# 6. Save the whole pipeline (preprocessing + model) as one file
# ---------------------------------------------------------------------
joblib.dump(pipeline, MODEL_OUT_PATH)
print(f"\nSaved trained pipeline to {MODEL_OUT_PATH}")
