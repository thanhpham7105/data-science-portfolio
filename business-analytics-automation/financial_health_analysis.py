#!/usr/bin/env python

# coding: utf-8

# Step 1: Import the pandas library
import pandas as pd

# Step 2: LOAD the business dataset into a DataFrame
df = pd.read_excel("business_dataset.xlsx")

# Step 3: Identify duplicate rows
duplicates = df[df.duplicated()]
if not duplicates.empty:
    print("\nDuplicate rows found:")
    print(duplicates)
else:
    print("\nNo duplicate rows found.")

# Step 4: Group by state and run descriptive stats
numeric_cols = df.select_dtypes(include='number').columns
grouped_stats = df.groupby('Business State')[numeric_cols].agg(['mean', 'median', 'min', 'max'])
grouped_stats_df = grouped_stats.reset_index()
print("Descriptive statistics grouped by state:")
print(grouped_stats_df.head())

# Step 5: Filter negative debt-to-equity ratios
if "Debt to Equity" in df.columns:
    negative_de = df[df["Debt to Equity"] < 0]
    print("\nBusinesses with Negative Debt to Equity:")
    print(negative_de)
else:
    print("\nColumn 'Debt to Equity' not found.")

# Step 6: Calculate debt-to-income ratio (Total Long-term Debt / Total Revenue) with divide-by-zero protection
def calculate_debt_to_income(debt, revenue):
    if revenue == 0:
        return 0 if debt == 0 else 1  # 0% if debt = 0, else 100%
    else:
        return debt / revenue

if "Total Long-term Debt" in df.columns and "Total Revenue" in df.columns:
    df["Debt to Income Ratio"] = df.apply(
        lambda row: calculate_debt_to_income(row["Total Long-term Debt"], row["Total Revenue"]),
        axis=1
    )
    print("\nDebt to Income Ratio added to DataFrame.")
else:
    print("\nRequired columns not found for Debt to Income Ratio.")

# Step 7: Show the updated DataFrame preview
print("\nPreview of updated DataFrame:")
print(df.head())
