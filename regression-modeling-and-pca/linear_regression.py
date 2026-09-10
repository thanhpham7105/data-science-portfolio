# Linear Regression: Modeling Housing Sale Price
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

housing_data = pd.read_csv("housing_data.csv")
# Rename Varibles for Clarity
column_map = {
    "Price": "Price",
    "SquareFootage": "SquareFeet",
    "NumBathrooms": "Bathrooms",
    "NumBedrooms": "Bedrooms",
    "DistanceToCityCenter": "Distance",
    "SchoolRating": "SchoolRating"
}
housing_df = housing_data[list(column_map.keys())].copy()
housing_df.rename(columns=column_map, inplace=True)

# C2. Descriptive Statistics
print("Descriptive Statistics")
summary_stats = housing_df.describe().T
print(summary_stats)
summary_stats.to_csv("housing_summary.csv")
# c3. Univariate (distribution of single variables)
for column in ["Price", "SquareFeet", "Bathrooms", "Bedrooms", "Distance", "SchoolRating"]:
    plt.figure(figsize=(6,4))
    sns.histplot(housing_df[column], kde=True, bins=20, color="skyblue")
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.savefig(f"unvariate {column}.png")
    plt.close()
# Bivariate (Price vs predictors)
for feature in ["SquareFeet", "Bathrooms", "Distance", "SchoolRating"]:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=housing_df[feature], y=housing_df["Price"], alpha=0.6)
    plt.title(f"Price vs {feature}")
    plt.tight_layout()
    plt.savefig(f"bivariate_price_vs_{feature}.png")
    plt.close()
# Discrete predictor (Bedrooms, bathrooms)
for feature in ["Bedrooms"]:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=housing_df, x=feature, y="Price", hue=feature, palette="muted", legend=False)
    plt.title(f"Price by {feature}")
    plt.xlabel(feature)
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(f"bivariate_price_vs_{feature}.png")
    plt.close()

#D1.Train / Test split
    feature = housing_df[["SquareFeet", "Bathrooms", "Bedrooms", "Distance", "SchoolRating"]]
    target_price = housing_df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        feature, target_price, test_size=0.25, random_state=42
    )
    print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# Save Training and test datasets
train_data = X_train.copy()
train_data["Price"] = y_train
test_data = X_test.copy()
test_data["Price"] = y_test
train_data.to_csv("train_housing.csv", index=False)
test_data.to_csv("test_housing.csv", index=False)

#D2. Regression Model D2
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)
linear_model = sm.OLS(y_train, X_train_const).fit()
print("Regression Model Summary")
print(linear_model.summary())

#D3. Training error
train_predictions = linear_model.predict(X_train_const)
train_mse = mean_squared_error(y_train, train_predictions)
print(f"Training MSE: {train_mse:,.2f}")

#D4. Test error
test_predictions = linear_model.predict(X_test_const)
test_mse = mean_squared_error(y_test, test_predictions)
print(f"Test MSE: {test_mse:,.2f}")

