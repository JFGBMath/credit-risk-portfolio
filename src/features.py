import pandas as pd
import numpy as np


def clean_term(df):
    """
    Converts term from string to integer.
    Example: ' 36 months' -> 36
    """
    df["term"] = df["term"].str.extract(r"(\d+)").astype(int)
    return df


def clean_emp_length(df):
    """
    Converts employment length from string to numeric.
    Imputes nulls with median.
    Example: '10+ years' -> 10, '< 1 year' -> 0
    """
    mapping = {
        "< 1 year": 0,
        "1 year": 1, "2 years": 2, "3 years": 3,
        "4 years": 4, "5 years": 5, "6 years": 6,
        "7 years": 7, "8 years": 8, "9 years": 9,
        "10+ years": 10
    }
    df["emp_length"] = df["emp_length"].map(mapping)
    df["emp_length"] = df["emp_length"].fillna(df["emp_length"].median())
    return df


def clean_grade(df):
    """
    Converts letter grade to numeric score.
    A=7 (best) to G=1 (worst).
    """
    grade_map = {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1}
    df["grade_num"] = df["grade"].map(grade_map)
    df = df.drop(columns=["grade", "sub_grade"])
    return df


def clean_categoricals(df):
    """
    One-hot encodes remaining categorical columns.
    """
    cat_cols = ["home_ownership", "verification_status", "purpose"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    return df


def impute_nulls(df):
    """
    Imputes remaining null values with column median.
    """
    for col in ["dti", "revol_util"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def create_financial_ratios(df):
    """
    Creates new features based on financial domain knowledge.
    Clips extreme values to avoid inf from zero division.
    """
    annual_inc_safe = df["annual_inc"].replace(0, np.nan)
    df["installment_to_income"] = df["installment"] / (annual_inc_safe / 12)
    df["loan_to_income"] = df["loan_amnt"] / annual_inc_safe
    df["installment_to_income"] = df["installment_to_income"].fillna(df["installment_to_income"].median())
    df["loan_to_income"] = df["loan_to_income"].fillna(df["loan_to_income"].median())
    df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df = df.drop(columns=["fico_range_low", "fico_range_high"])
    return df


def merge_macro_features(loans_df, macro_df):
    """
    Joins macroeconomic indicators to loans by issue date.
    Each loan gets the macro values from its origination month.
    """
    loans_df["issue_d"] = pd.to_datetime(loans_df["issue_d"], format="%b-%Y")
    macro_df.index = pd.to_datetime(macro_df.index)

    loans_df = loans_df.merge(
        macro_df,
        left_on="issue_d",
        right_index=True,
        how="left"
    )
    loans_df = loans_df.drop(columns=["issue_d"])
    return loans_df


def build_features(loans_df, macro_df):
    """
    Master function — runs the full feature engineering pipeline.
    """
    print("Building features...")

    loans_df = clean_term(loans_df)
    print("  -> term cleaned")

    loans_df = clean_emp_length(loans_df)
    print("  -> emp_length cleaned")

    loans_df = clean_grade(loans_df)
    print("  -> grade encoded")

    loans_df = impute_nulls(loans_df)
    print("  -> nulls imputed")

    loans_df = create_financial_ratios(loans_df)
    print("  -> financial ratios created")

    loans_df = merge_macro_features(loans_df, macro_df)
    print("  -> macro features merged")

    loans_df = clean_categoricals(loans_df)
    print("  -> categoricals encoded")

    print(f"\nFinal dataset: {loans_df.shape[0]:,} rows, {loans_df.shape[1]} columns")
    print(f"Remaining nulls: {loans_df.isnull().sum().sum()}")

    return loans_df


if __name__ == "__main__":
    import os

    loans = pd.read_csv("data/processed/loans_clean.csv")
    macro = pd.read_csv("data/processed/macro_indicators.csv", index_col="date")

    features_df = build_features(loans, macro)

    os.makedirs("data/processed", exist_ok=True)
    features_df.to_csv("data/processed/features.csv", index=False)
    print("\nSaved to data/processed/features.csv")