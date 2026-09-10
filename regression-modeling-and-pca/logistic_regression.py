# ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load Dataset
housing_data = pd.read_csv("housing_data.csv")
# Rename Variables for Clarity
column_map = {
    "Price": "Price",
    "SquareFootage": "SquareFeet",
    "NumBathrooms": "Bathrooms",
    "NumBedrooms": "Bedrooms",
    "PreviousSalePrice": "PrevPrice",
    "Garage": "Garage",
    "Fireplace": "Fireplace",
    "IsLuxury": "Luxury"
}
housing_df = housing_data[list(column_map.keys())].copy()
housing_df.rename(columns=column_map, inplace=True)

#C2. Descriptive Statistics
print("Descriptive Stastics")
summary_stats = housing_df.describe(include="all").T
print(summary_stats)
summary_stats.to_csv("housing_logit_summary.csv")

#C3. Univarite Visualizations
for column in ["Luxury", "Price", "SquareFeet", "Bathrooms", "Bedrooms", "PrevPrice","Garage", "Fireplace"]:
    plt.figure(figsize=(6,4))
    sns.histplot(housing_df[column], kde=True, bins=20, color="skyblue")
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.savefig(f"univariate_{column}.png")
    plt.close()

# Bivariate Visualizations (Luxury vs Predictors)
for feature in ["Price", "SquareFeet", "Bathrooms", "Bedrooms", "PrevPrice"]:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="Luxury", y=feature, data=housing_df)
    plt.title(f"{feature} vs Luxury Status")
    plt.tight_layout()
    plt.savefig(f"Bivariate_{feature}_vs_Luxury.png")
    plt.close()

for cat in ["Garage", "Fireplace"]:
    plt.figure(figsize=(6,4))
    sns.countplot(x=cat, hue="Luxury", data=housing_df)
    plt.title(f"{cat} vs Luxury")
    plt.tight_layout()
    plt.savefig(f"bivariate_{cat})_vs_Luxury.png")
    plt.close()

#D1. Train/Test Split
features = housing_df.drop("Luxury", axis=1)
target = housing_df["Luxury"]
features = pd.get_dummies(features, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    features,target, test_size=0.3, stratify=target, random_state=42
)
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# Save Training and Test Datasets
train_data = X_train.copy()
train_data["Luxury"] = y_train
test_data = X_test.copy()
test_data.to_csv("train_housing_logit.csv", index=False)
test_data.to_csv("test_housing_logit.csv", index=False)
# Ensure all features and target are numeric
X_train = X_train.apply(pd.to_numeric, errors="coerce").astype(float)
y_train = pd.to_numeric(y_train, errors="coerce").astype(float)

# Drop row with NaN created during coercion
mask = X_train.notna().all(axis=1) & y_train.notna()
X_train = X_train.loc[mask]
Y_train = y_train.loc[mask]

# D2. Logistic Regression Model with Backward Elimination
def backward_elimination(X, y, alpha=0.05):
    variables = X.columns.tolist()
    while True:
        model = sm.Logit(y, sm.add_constant(X[variables])).fit(disp=False)
        p_values = model.pvalues.drop("const")
        if p_values.max() > alpha:
            drop_var = p_values.idxmax()
            print(f"Removing {drop_var} (p = {p_values.max():.4f})")
            variables.remove(drop_var)
        else:
            return variables, model

final_vars, logit_model = backward_elimination(X_train, y_train)
print("Final Predictors:", final_vars)
print("Logistic Regression Summary")
print(logit_model.summary())

#D3. Training Cofusion Matrix and Accuracy
train_pred = (logit_model.predict(sm.add_constant(X_train[final_vars])) > 0.5).astype(int)
train_cm = confusion_matrix(y_train, train_pred)
train_acc = accuracy_score(y_train, train_pred)
print("Training Confusion Matrix:", train_cm)
print(f"Training Accuracy: {train_acc:.3f}")

#D4. Test Confusion Matrix and Accuracy
test_pred = (logit_model.predict(sm.add_constant(X_test[final_vars])) > 0.5).astype(int)
test_cm = confusion_matrix(y_test, test_pred)
test_acc = accuracy_score(y_test, test_pred)
print("Test Confusion Matrix:", test_cm)
print(f"Test Accuracy: {test_acc:.3f}")

# E5. Logistic Regression Assumptions Check
# Multicollinearity (VIF)
X_const = sm.add_constant(X_train[final_vars])
vif_table = pd.DataFrame({
    "Variale": X_const.columns,
    "VIF": [variance_inflation_factor(X_const.values, i)
            for i in range(X_const.shape[1])]
})
print("VIF Table", vif_table)
vif_table.to_csv("vif_table.csv", index=False)

# Influential Outliers (Cook's Distance)
influence = logit_model.get_influence()
cooks_d = influence.cooks_distance[0]
plt.figure(figsize=(6, 4))
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
plt.axhline(4/len(X_train), color="red", linestyle="--")
plt.title("Cook's Distance for Influential Observations")
plt.tight_layout()
plt.savefig("cooks_distance_png")
plt.close()

