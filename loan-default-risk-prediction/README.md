# Loan Default Risk Prediction

## Overview
A predictive modeling project assessing which borrower and loan characteristics most influence loan default, aimed at helping lenders move from rule-of-thumb underwriting toward a data-driven, standardized risk assessment.

**Research question:** To what extent do borrower income, credit score, loan amount, and debt-to-income ratio (DTI) affect a loan's final outcome (default vs. paid)?

**Hypotheses tested:**
- H0: Income, credit score, loan amount, and DTI have no statistically significant effect on loan outcome.
- H1: Income, credit score, loan amount, and DTI do have a statistically significant effect on loan outcome.

## What's here
- `loan_default_risk_model.py` -- end-to-end modeling script: data extraction/validation, a preprocessing pipeline (imputation + scaling via `ColumnTransformer`), and a comparison of logistic regression, decision tree, and random forest classifiers, evaluated with accuracy, precision, recall, F1, and ROC-AUC.

## Approach
- Exploratory analysis of borrower income, credit score, loan amount, and DTI against loan outcome.
- A scikit-learn `Pipeline`/`ColumnTransformer` to handle missing values and scaling consistently across train/test splits.
- Three classifier families compared head-to-head (logistic regression as a baseline, plus tree-based models) on the same held-out test set.
- Findings translated into recommendations for underwriting policy and risk-based capital allocation.

## Business value
A model that flags high-risk applicants earlier lets lenders adjust underwriting policy and allocate capital more effectively, while reducing losses from default and manual-review costs.

## Skills demonstrated
End-to-end project scoping (research question, hypotheses, ethical screening), scikit-learn pipelines, multi-model comparison for classification, applied machine learning for credit risk, stakeholder-ready reporting.
