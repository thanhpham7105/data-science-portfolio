# Regression Modeling & PCA

## Overview
Built and evaluated three predictive models on a real-estate dataset to answer questions about what drives property pricing and classification.

## 1. Linear regression
Multiple linear regression to identify which housing characteristics most influence sale price.
- `linear_regression.py`

## 2. Logistic regression
Classification model (with backward elimination for feature selection) to predict whether a property qualifies as "luxury," evaluated via confusion matrix and accuracy on held-out data, with a full assumptions check.
- `logistic_regression.py`

## 3. PCA regression
Principal Component Analysis (standardization, component selection via the elbow rule, variance-explained review) feeding into a regression model, as a dimensionality-reduction comparison against the direct-feature model.
- `pca_regression.py`

## Skills demonstrated
Linear & logistic regression, model diagnostics/assumption checking, train/test evaluation, PCA for dimensionality reduction.
