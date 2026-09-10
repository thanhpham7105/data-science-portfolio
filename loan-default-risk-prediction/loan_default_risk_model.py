# ============================================
# Loan Default Risk -- Predictive Modeling
# Predictors: income, credit score, loan amount, debt-to-income ratio -> Status
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------
# 1) DATA EXTRACTION
# --------------------------------------------
df = pd.read_csv("loan_data.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumns:")
print(list(df.columns))
print("\nInfo:")
print(df.info())

# --------------------------------------------
# 2) DEFINE VARIABLES 
# --------------------------------------------
X_cols = ["income", "Credit_Score", "loan_amount", "dtir1"]
y_col = "Status"

missing_cols = [c for c in X_cols + [y_col] if c not in df.columns]
if missing_cols:
    raise ValueError(f"These required columns are missing from the dataset: {missing_cols}")

X = df[X_cols].copy()
y = df[y_col].copy()

print("VARIABLE CONFIRMATION")
print("Predictors (X):", X_cols)
print("Target (y):", y_col)

# --------------------------------------------
# 3) DATA PREPARATION (missingness + types)
# --------------------------------------------
print("MISSING VALUES CHECK")
print("Missing values in predictors:")
print(X.isna().sum())
print("\nMissing values in target:")
print(y.isna().sum())

# Ensure target is usable
if not pd.api.types.is_numeric_dtype(y):
    raise ValueError(
        "Status is not numeric. If Status is text (e.g., Default/Paid), map it to 1/0 before modeling."
    )

unique_status = sorted(pd.Series(y.dropna().unique()).tolist())
print("\nUnique Status values:", unique_status)
if not set(unique_status).issubset({0, 1}):
    raise ValueError(f"Status must be binary 0/1 for this code. Found: {unique_status}")

# Convert predictors to numeric safely (coerce errors to NaN)
for c in X_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")

print("SUMMARY STATS")
print(X.describe())

# Optional: quick distributions (good screenshots)
for col in X_cols:
    plt.figure()
    X[col].hist(bins=30)
    plt.title(f"Distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# --------------------------------------------
# 4) TRAIN/TEST SPLIT
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("TRAIN/TEST SPLIT")
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("Train Status rate:", y_train.mean(), "Test Status rate:", y_test.mean())

# --------------------------------------------
# FINAL PREPARED DATASET (for Section C evidence)
# --------------------------------------------
df_modeling = df[["income", "Credit_Score", "loan_amount", "dtir1", "Status"]].copy()

print("Final prepared dataset shape:")
print(df_modeling.shape)

print("\nFinal prepared dataset:")
print(df_modeling.head())

# --------------------------------------------
# 5) PREPROCESSING PIPELINE
#   - Median imputation for missing numeric values
#   - Standardization for logistic regression
# --------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocess = ColumnTransformer(
    transformers=[("num", numeric_transformer, X_cols)],
    remainder="drop"
)

# --------------------------------------------
# 6) MODEL 1: LOGISTIC REGRESSION (interpretability: "extent")
# --------------------------------------------
log_reg = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

log_reg.fit(X_train, y_train)

lr_proba = log_reg.predict_proba(X_test)[:, 1]
lr_pred = (lr_proba >= 0.50).astype(int)

print("\n===== LOGISTIC REGRESSION RESULTS (threshold=0.50) =====")
print("ROC AUC:", roc_auc_score(y_test, lr_proba))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_pred))
print("\nClassification Report:")
print(classification_report(y_test, lr_pred, digits=4))

plt.figure()
RocCurveDisplay.from_predictions(y_test, lr_proba)
plt.title("ROC Curve — Logistic Regression")
plt.tight_layout()
plt.show()

# "To what extent" output: coefficients + odds ratios
lr_model = log_reg.named_steps["model"]
coef = pd.Series(lr_model.coef_[0], index=X_cols).sort_values()
odds_ratio = np.exp(coef)

print("\n===== EXTENT / IMPACT (Logistic Regression) =====")
print("Coefficients (direction & strength):")
print(coef)
print("\nOdds Ratios:")
print(odds_ratio)

# --------------------------------------------
# 7) MODEL 2: DECISION TREE (nonlinear comparison)
# --------------------------------------------
tree = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", DecisionTreeClassifier(
        random_state=42,
        max_depth=5,
        min_samples_leaf=50
    ))
])

tree.fit(X_train, y_train)

tree_proba = tree.predict_proba(X_test)[:, 1]
tree_pred = (tree_proba >= 0.50).astype(int)

print("\n===== DECISION TREE RESULTS (threshold=0.50) =====")
print("ROC AUC:", roc_auc_score(y_test, tree_proba))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, tree_pred))
print("\nClassification Report:")
print(classification_report(y_test, tree_pred, digits=4))

plt.figure()
RocCurveDisplay.from_predictions(y_test, tree_proba)
plt.title("ROC Curve — Decision Tree")
plt.tight_layout()
plt.show()

# --------------------------------------------
# 8) MODEL 3: RANDOM FOREST (approved in form; stronger performance)
# --------------------------------------------
rf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    ))
])

rf.fit(X_train, y_train)

rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred = (rf_proba >= 0.50).astype(int)

print("\n===== RANDOM FOREST RESULTS (threshold=0.50) =====")
print("ROC AUC:", roc_auc_score(y_test, rf_proba))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, digits=4))

plt.figure()
RocCurveDisplay.from_predictions(y_test, rf_proba)
plt.title("ROC Curve — Random Forest")
plt.tight_layout()
plt.show()

# Feature importance (Random Forest)
rf_model = rf.named_steps["model"]
importances = pd.Series(rf_model.feature_importances_, index=X_cols).sort_values(ascending=False)

print("\n===== RANDOM FOREST FEATURE IMPORTANCE =====")
print(importances)

plt.figure()
importances.plot(kind="bar")
plt.title("Random Forest Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# --------------------------------------------
# 9) Summary table of model metrics (good for report)
# --------------------------------------------
def metric_row(name, y_true, y_pred, y_proba):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba)
    }

summary = pd.DataFrame([
    metric_row("Logistic Regression", y_test, lr_pred, lr_proba),
    metric_row("Decision Tree", y_test, tree_pred, tree_proba),
    metric_row("Random Forest", y_test, rf_pred, rf_proba)
])

print("\n===== MODEL COMPARISON SUMMARY =====")
print(summary)

# --------------------------------------------
# 10) Save modeling dataset used (optional evidence)
# --------------------------------------------
df_modeling = df[X_cols + [y_col]].copy()
df_modeling.to_csv("modeling_dataset.csv", index=False)
print("\nSaved file: modeling_dataset.csv")
