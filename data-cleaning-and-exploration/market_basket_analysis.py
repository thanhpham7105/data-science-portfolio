import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
df = pd.read_csv("retail_transactions_raw.csv")

df.columns = df.columns.str.strip()
print("Columns in dataset:", df.columns.tolist())

# Part C1: Data Wrangling & Encoding
# C1b. Encoding
# Order Priority: Medium < High
if "OrderPriority" in df.columns:
    priority_mapping = {"Medium": 1, "High":2}
    df["OrderPrority_encoded"] = df["OrderPriority"].map(priority_mapping)
else:
    print("Column 'OrderPriority' not found, skipping encoding.")

# Customer Satisfaction: Linkert scale already numeric (1-5)
if "CustomerOrderSatisfaction" in df.columns:
    satisfaction_mapping = {"Very Dissatisfied": 1,
                            "Dissatisfied": 2,
                            "Prefer not to answer": 3,
                            "Satisfied": 4,
                            "Very Satisfied": 5}
    df["CostumerSatisfaction_Encoded"] = df["CustomerOrderSatisfaction"].map(satisfaction_mapping)
else:
    print("Column ' CustomerOrderSatisfaction' not , skipping ordinal encoding.")

# One hot Encoding
# Payment Method
if "PaymentMethod" in df.columns:
    payment_dummies = pd.get_dummies(df["PaymentMethod"], prefix="PaymentMethod").astype(int)
    df = pd.concat([df, payment_dummies], axis=1)
else:
    print("Column 'PaymentMethod' not found, skipping one-hot encoding.")

# Region (Nominal Variable)
if "Region" in df.columns:
    region_dummies = pd.get_dummies(df["Region"], prefix= "Region").astype(int)
    df = pd.concat([df, region_dummies], axis=1)
else:
    print("Column 'Region' not found, skipping one-hot encoding.")

# Export the encoded datset fot future use
df.to_csv("megasore_encoded.csv", index=False)
print(" Encoded dataset saved as 'retail_encoded.csv'")

# Part C2: Transactionalization
# Ensure required column exist
if not {"OrderID", "ProductName"}. issubset(df.columns):
    raise KeyError("Columns 'OrderID' and/or ' ProductName' missing in dataset")

# Group products by orderid to create transaction lists
transactional_df = df.groupby("OrderID")["ProductName"].apply(list).reset_index()
transactional_df.to_csv("retail_transactions.csv", index=False)
print("Transactionalized dataset exported as retail_transactions.csv")

# Part C2c: Appriori Analysis
basket = pd.get_dummies(
    df[["OrderID", "ProductName"]].groupby("OrderID")["ProductName"].apply(list).explode()

).groupby(level=0).max()

# Apply Apriori
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)

# Generate rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Sort and select top 3 rules by lift
top_rules = rules.sort_values("lift", ascending=False).head(3)
print("Top 3 Association Rules sorted by lift:")
print(top_rules[["antecedents", "consequents", "support", "confidence", "lift"]])
