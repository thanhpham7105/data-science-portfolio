import pandas as pd
import numpy as np
df = pd.read_csv("Employee Turnover Dataset.csv")
number_rows = df.shape[0]
number_colums = df.shape[1]
print(f"Number of Rows:{number_rows}")
print(f"Number of colums:{number_colums}")
df.columns = df.columns.str.strip()
print("Data Inspection")
duplicated_count = df.duplicated().sum()
print(f"Duplicated:{duplicated_count}")
Missing_values = df.isnull().sum()
Missing_values = Missing_values[Missing_values>0]
print(f"Missing Values:{Missing_values}")
Catagorical_colums = ["Gender", "MaritalStatus","Turnover", "PaycheckMethod", "CompensationType"]
for col in Catagorical_colums:
    if col in df.columns:
        print(f"{col}: {df[col].unique()}")
#4. Formatting Error
#Check column name with space
Column_with_space = [col for col in df.columns if col.strip()!= col]
print(f"Column Names with space:{Column_with_space}")
#Check for leading or trailing in string Values
string_columns = df.select_dtypes(include="object").columns
for col in string_columns:
    if df[col].str.contains(r'^\s|\s$', regex=True).any():
        print(f" Formatting error in value of column:{col}")
#5. Outlier detection using IQR
def detect_outliers_iqr(dataframe):
    result= {}
    for col in dataframe.select_dtypes(include=[np.number]).columns:
        q1 = dataframe[col].quantile(0.25)
        q3 = dataframe[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = dataframe[(dataframe[col] < lower) | (dataframe[col] > upper)]
        result[col] = {
            "outlier_count": len(outliers) ,
            "lower_bound": lower,
            "upper_bound": upper
        }
        return pd.DataFrame(result).T
outlier_summary = detect_outliers_iqr(df)
print("5. Outlier summary ( IQR medthod):")
print(outlier_summary)

#C1: Data Cleaning
#1. Remove Duplicate Rows
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicates removed: {duplicated_count}")
#2. Fill Missing Values
missing_before = df.isnull().sum().sum()
for col in df.columns:
    if df[col].isnull().sum() == 0:
        continue
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col]= df[col].fillna(df[col].median())
    else:
        mode_val = df[col].mode(dropna=True)
        fill_val = mode_val[0]if not mode_val.empty else "Unkown"
        df[col] = df [col].fillna(fill_val)
missing_after = df.isnull().sum().sum()
print(f"Missing Value filled: {missing_before} to {missing_after}")
#3. Normalize Categorial Values
columns_to_title =["Gender", "MaritalStatus", "CompenstationType", "PaycheckMethod", "JobRole"]
for col in columns_to_title:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

#Convert Yes/No values
def normalize_yn(val):
    mapping = {
        "yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes",
        "no": "No", "n":"No", "false": "No", "0": "No"
    }
    val_clean= str(val).strip().lower()
    return mapping.get(val_clean, str(val).strip().title())
for col in ["Turnover", "TextMessageOptIn"]:
    if col in df.columns:
        df[col] = df[col].apply(normalize_yn)
print("Catagorial Value standardized and Yes/No Normalized")
#4. Clean string and Monetary Columns
for col in string_columns:
    df[col] = df[col].astype(str).str.strip()
#Remove $ and , from Monetary columns
monetary_columns = ["HourlyRate", "AnnualSalary"]
for col in monetary_columns:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace(r'[\$,]', '', regex=True)
            .str.replace(",", "")
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

#Convert other numeric columns
numeric_to_convert = [
    "Age", "Tenure", "HoursWeekly", "DrivingCommuterDistance",
    "NumCompaniesPreviousWorked", "AnnualProfessionalDevHrs"
]
for col in numeric_to_convert:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors= "coerce")
#5. Winsorize Outliers
for col in df.select_dtypes(include=[np.number]).columns:
    q1 = df[col]. quantile(0.25)
    q3 = df[col]. quantile(0.75)
    iqr = q3-q1
    lower = q1 - 1.5* iqr
    upper = q3 + 1.5 * iqr
    df[col] = df[col].clip(lower, upper)
print("Outliers capped to IQR bounds")
# Save cleaned dataset
df.to_csv("Employee_Turnover_Clean.csv", index=False)
print("Cleaned dataset saved as: Employee_Turnover_Clean.csv")