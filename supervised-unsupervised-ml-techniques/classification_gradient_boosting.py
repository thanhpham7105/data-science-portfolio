import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# D. Data Preparation
churn_df = pd.read_csv("churn_clean.csv")
print("Loaded dataset shape:", churn_df.shape)
features_to_keep = [
    "Churn","Tenure","MonthlyCharge","Bandwidth_GB_Year",
    "Outage_sec_perweek","Email","Contacts","Yearly_equip_failure",
    "Age","Children","Income","Gender","Marital","InternetService",
    "Phone","Multiple","OnlineSecurity","OnlineBackup","DeviceProtection",
    "TechSupport","StreamingTV","StreamingMovies","PaperlessBilling",
    "PaymentMethod","Contract","Techie",
    "Item1","Item2","Item3","Item4","Item5","Item6","Item7","Item8"
]

churn_df_model = churn_df[features_to_keep].copy()

# Handle missing values
numeric_features    = churn_df_model.select_dtypes(include=["int64","float64"]).columns
categorical_features = churn_df_model.select_dtypes(include=["object"]).columns

churn_df_model[numeric_features]    = churn_df_model[numeric_features].fillna(churn_df_model[numeric_features].median())
churn_df_model[categorical_features] = churn_df_model[categorical_features].fillna(churn_df_model[categorical_features].mode().iloc[0])

# Encode yes/no categorical variables into numeric values
binary_map = {"yes":1, "no":0, "Yes":1, "No":0}
binary_vars = [
    "Phone","Multiple","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV",
    "StreamingMovies","PaperlessBilling","Techie","Churn"
]

for col in binary_vars:
    churn_df_model[col] = churn_df_model[col].map(binary_map)

# One-hot encode remaining categorical variables
churn_df_encoded = pd.get_dummies(churn_df_model, drop_first=True)
print("Encoded dataset shape:", churn_df_encoded.shape)
# Save cleaned data
churn_df_encoded.to_csv("churn_clean_prepared.csv", index=False)

# E1. Split into Train / Validation / Test (60/20/20)
X = churn_df_encoded.drop(columns=["Churn"])
y = churn_df_encoded["Churn"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
X_train.to_csv("X_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)

X_val.to_csv("X_val.csv", index=False)
y_val.to_csv("y_val.csv", index=False)

X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# E2. Train initial Gradient Boosting model
initial_model = GradientBoostingClassifier(random_state=42)
initial_model.fit(X_train, y_train)

y_train_pred = initial_model.predict(X_train)
y_train_prob = initial_model.predict_proba(X_train)[:, 1]

print("Initial Model")
print("Accuracy: ", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred))
print("Recall:   ", recall_score(y_train, y_train_pred))
print("F1 Score: ", f1_score(y_train, y_train_pred))
print("AUC-ROC:  ", roc_auc_score(y_train, y_train_prob))
print("Confusion Matrix:", confusion_matrix(y_train, y_train_pred))

# E3. Hyperparameter Tuning (k-fold CV)
param_grid = {
    "n_estimators": [100, 150],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 3],
}

grid_search_model = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
)

print("\nTuning Gradient Boost parameters")
grid_search_model.fit(X_train, y_train)

print("Best Parameters:", grid_search_model.best_params_)
print("Best Cross-Validated AUC:", grid_search_model.best_score_)

# E4. Evaluate Optimized Model on Test Set
best_model = grid_search_model.best_estimator_
y_test_pred = best_model.predict(X_test)
y_test_prob = best_model.predict_proba(X_test)[:, 1]

print("Optimized Model")
print("Accuracy:  ", accuracy_score(y_test, y_test_pred))
print("Precision: ", precision_score(y_test, y_test_pred))
print("Recall:    ", recall_score(y_test, y_test_pred))
print("F1 Score:  ", f1_score(y_test, y_test_pred))
print("AUC-ROC:   ", roc_auc_score(y_test, y_test_prob))
print("Confusion Matrix:", confusion_matrix(y_test, y_test_pred))
