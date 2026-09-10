
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score
import statsmodels.api as sm

# Loading Dataset
housing_df = pd.read_csv("housing_data.csv")
print(housing_df.shape)

# D1. Select Countinous Variables
countinous_vars = [
    'SquareFootage', 'NumBathrooms', 'NumBedrooms', 'BackyardSpace',
    'CrimeRate', 'SchoolRating', 'AgeOfHome', 'DistanceToCityCenter',
    'EmploymentRate', 'PropertyTaxRate', 'RenovationQuality',
    'LocalAmenities', 'TransportAccess', 'Floors', 'Windows', 'PreviousSalePrice'
]
housing_cont = housing_df[countinous_vars]

# D2. Standardize Continous Variables
scaler = StandardScaler()
housing_scaled = pd.DataFrame(scaler.fit_transform(housing_cont), columns=countinous_vars)
housing_scaled.to_csv("housing_cleaned_standardized.csv")
# Save Desciptive Statistics
desc_before = housing_df[countinous_vars + ["Price"]].describe().T
desc_after = housing_scaled.describe().T
desc_before.to_csv("housing_descriptive_before_standardization.csv")
desc_after.to_csv("housing_descriptive_after_standardization.csv")

#E1. Perform PCA
pca = PCA()
housing_pca = pca.fit_transform(housing_scaled)
pca_df = pd.DataFrame(housing_pca, columns=[f'PC{i+1}'for i in range(housing_scaled.shape[1])])

#E2. Dertermine Number of Principal Components (Elbow Rule)
explained_variance = pca.explained_variance_ratio_
cum_variance = explained_variance.cumsum()
# Scree Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance)+1), explained_variance, 'o-', label="Individual variane")
plt.plot(range(1, len(cum_variance) + 1), cum_variance, 's--', label="Cumulative variance")
plt.axvline(x=5, color='red', linestyle='--', label='Elbow at PC5')
plt.title("Scree Plot - PCA on Housing Data")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.legend()
plt.tight_layout()
plt.savefig("pca_scree_plot.png", dpi=300)
plt.show()

#E3. Variance Summary
pca_variance_df = pd.DataFrame({
    "Principal Component": [f"PC{i+1}" for i in range(len(explained_variance))],
    'Explained Variance (%)': np.round(explained_variance * 100, 2),
    'Cumulative Variance (%)' :np.round(cum_variance * 100, 2)
})
print(" PCA Variance Summary (First 5 Components")
print(pca_variance_df.head(5).round(4))

# Extract top 3 positive and top 3 negative contributors for each PC1-PC5
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(pca.components_))],
    index=housing_scaled.columns
)
summary_rows = []
for i in range(5):
    pc = f'PC{i+1}'
    top_pos = loadings[pc].sort_values(ascending=False).head(3)
    top_neg = loadings[pc].sort_values(ascending=True).head(3)

    summary_rows.append({
        "Principal Component": pc,
        "Top Postive Variables": ", ".join(top_pos.index),
        "Top Negative Variables": ", ".join(top_neg.index)
    })
pca_summary_table = pd.DataFrame(summary_rows)
pca_summary_table.to_csv("pca_summary_table.csv", index=False)

#F1. Split Data
X= pca_df.iloc[:, :5]
y = housing_df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Save the training and test datasets
train_data = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
test_data = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)

train_data.to_csv("training_dataset.csv", index=False)
test_data.to_csv("test_dataset.csv", index=False)

#F2. Optimized Regression Model (Backward Elimination)
X_train_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_const).fit()
def backward_elimination(model, X, y, alpha=0.05):
    while True:
        pvalues = model.pvalues
        worst_pval = pvalues.drop("const").max()
        if worst_pval > alpha:
            drop_var = pvalues.drop("const").idxmax()
            print(f"Removing {drop_var} (p={worst_pval:.4f})")
            X = X.drop(columns=[drop_var])
            model = sm.OLS(y, X).fit()
        else:
            break
    return model, X.columns
optimized_model, final_features = backward_elimination(model, X_train_const, y_train)
print("Optimized Regression Model Summary")
print(optimized_model.summary())
coeffs = optimized_model.params
pvals = optimized_model.pvalues
for var, coef in coeffs.items():
    print(f"{var}: Coefficient = {coef:.2f}, p-value = {pvals[var]:.4f}")

#F3. Predict on training set using the optimized model
X_train_final = X_train_const[final_features]
y_train_pred = optimized_model.predict(X_train_final)
train_mse = mean_squared_error(y_train, y_train_pred)
print(f"F3: Mean Squared Error (Training Set): {train_mse:,.2f}")

#F4. Calculate MSE for the test set
X_test_final = sm.add_constant(X_test[final_features.drop('const', errors='ignore')])
y_test_pred = optimized_model.predict(X_test_final)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"F4: Mean Squares Error (Test Set): {test_mse:,.2f}")
