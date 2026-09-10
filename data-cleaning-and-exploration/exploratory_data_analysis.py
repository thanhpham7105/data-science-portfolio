import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency

sns.set(style="whitegrid")

FIG_DIR = os.path.join("output", "figures")
TAB_DIR = os.path.join("output", "table")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)
df = pd.read_csv("Health Insurance Dataset.csv")
# Part I: Unvivariate Analysis
# 1. Age: Histogram + KDE
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age" , kde= True, bins=30)
plt.title('Distribution of Age')
plt.xlabel('Age'); plt.ylabel('Frequency')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "univeraiate_age.png"), dpi=150)
plt.show()

#2. Charge: Histogram + KDE
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x= 'charges', kde=True, bins = 30 )
plt.title('Distribution of Insurance Charges')
plt.xlabel('Charges'); plt.ylabel('Frequency')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "univariate_charge.png"), dpi=150)
plt.show()

#3. Smoker: count plot
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='smoker')
plt.title('Count of Smokers')
plt.xlabel('Smoker'); plt.ylabel('Count')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "univariate_smoker.png"), dpi=150)
plt.show()

#4. Region count plot
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='region')
plt.title('Count by Region')
plt.xlabel('Region'); plt.ylabel('Count')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "univariate_region.png"), dpi=150)
plt.show()

#Part I: A2. Bivariate Analysis
# Age: age vs charges (scatter), age by region(box)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='age', y='charges', hue='smoker')
plt.title('Age vs Charges by Smoker Status')
plt.xlabel('Age'); plt.ylabel('Charges')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "bivariate_age_charges.png"), dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='region', y='age')
plt.title('Age Distribution by Region')
plt.xlabel('Region'); plt.ylabel('Age')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "binavariate_age_region.png"), dpi=150)
plt.show()

# Charges: charges vs smoker(box), charges vs region (violin)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='smoker', y='charges')
plt.title('Charges by Smoking Statues')
plt.xlabel('Smoker'); plt.ylabel('Charges')
plt.tight_layout() ; plt.savefig(os.path.join(FIG_DIR, "bivariate_charges_smoker.png"), dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x='region', y='charges')
plt.title('Charges by Region')
plt.xlabel('Region'); plt.ylabel('Charges')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "bivariate_region_charge.png"), dpi=150)

# Smoker: smoker vs charges (box), smoker vs age (violin)
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='smoker', y='charges')
plt.title('Charges by Smoking status')
plt.xlabel('Smoker'); plt.ylabel('Charges')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "bivariate_smoker_charges.png"), dpi=150)

plt.figure(figsize= (8,5))
sns.violinplot(data=df, x='smoker', y='age')
plt.title('Age Distribution by Smoker status')
plt.xlabel('Smoker'); plt.ylabel('Age')
plt.tight_layout; plt.savefig(os.path.join(FIG_DIR, "bivariate_smoker_age.png"), dpi=150)
plt.show()

# Region: region vs charges (box), region vs smoker (stacked bar)
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='region', y='charges')
plt.title('Charges By Region')
plt.xlabel('Region'); plt.ylabel('Charges')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "bivariate_region_charges.png"), dpi=150)
plt.show()

plt.figure(figsize=(8,5))
region_smoker_ct = pd.crosstab(df['region'], df['smoker'])
region_smoker_ct.plot(kind='bar', stacked=True, figsize=(8,5))
plt.title('Smoker Count By Region')
plt.xlabel('Region'); plt.ylabel('Count')
plt.legend(title='Smoker')
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "bivariate_region_smoker.png"), dpi=150)
plt.show()

# Part I: B1.Desciptive Statistics
# Numeric summary (adds variance, rage, IQR)
num = df.select_dtypes(include=[np.number])
desc = num.describe(percentiles=[0.25, 0.5, 0.75]).T
desc.rename(columns={'50%': 'median'}, inplace=True)
desc['variance'] = num.var()
desc['range'] = num.max() - num.min()
desc['IQR'] = desc['75%'] - desc['25%']
print("Summary Statistics")
print(desc[['count', 'mean', 'median', 'std', 'variance', 'min', '25%', '75%', 'max', 'range', 'IQR']])
desc.to_csv(os.path.join(TAB_DIR, "quantitative_summary.csv"))

# b2. Categorical proportions
categorical = df.select_dtypes(include=['object'])
print("\nSummary Statistics (Categorical Variables)")
for col in categorical.columns:
    counts = df[col].value_counts()
    percentages = df[col].value_counts(normalize=True) * 100
    summary = pd.DataFrame({"Count": counts, "Percentage": percentages.round(2)})
    print(f"\n{col} Distribution:")
    print(summary)
    summary.to_csv(os.path.join(TAB_DIR, f"{col}_summary.csv"))

# Part II: Parametric Test ( Welch's t-test)
alpha = 0.05
charges_smokers = df[df['smoker'] == 'yes']['charges'].dropna()
charges_nonsmokers = df[df['smoker'] == 'no']['charges'].dropna()
group_summary = pd.DataFrame({
    "group": ["smoker_yes", "smoker_no"],
    "n": [charges_smokers.size, charges_nonsmokers.size],
    "mean": [charges_smokers.mean(), charges_nonsmokers.mean()],
    "std": [charges_smokers.std(ddof=1), charges_nonsmokers.std(ddof=1)]
})
group_summary.to_csv(os.path.join(TAB_DIR, "t-test_group_summary.csv"), index=False)

# Welch's t-test
t_stat, p_value = ttest_ind(charges_smokers, charges_nonsmokers, equal_var=False)
print("T-Test: Charge for Smokers vs Non_Smokers")
print(group_summary.to_string(index=False))
print(f"T-statistic:{t_stat:.4f}")
print(f"P-value: {p_value:.6f}")
if p_value < alpha:
    print("Conclusion: Significant difference in charges between smokers and non-smokers (Reject H0).")
else:
    print("Conclusion: No significant difference in charges between smokers and no-smokers (fail to reject H0).")

# Part III: Non-parametric test (Chi-Square)
contingency = pd.crosstab(df['region'], df['smoker'])
chi2, pvalue, df, expected = chi2_contingency(contingency)
print("Chi-Square Test: Region vs Smoking Status")
print("Contingency Table:")
print(contingency)
print("Expected Counts:")
print(pd.DataFrame(expected, index=contingency.index, columns=contingency.columns))
print(f"Chi-Square statistic: {chi2:.4f}")
print(f"Degrees of Freedom: {df}")
print(f"P-Value: {pvalue:.4f}")
if pvalue < alpha:
    print("Conclusion: Smoking status is dependent on region (Reject H0).")
else:
    print("Conclusion: No significant relationship between region and smoking status (Fail to reject H0).")