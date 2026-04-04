import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")


def get_fred_client():
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY not found in .env file")
    return Fred(api_key=FRED_API_KEY)



def fetch_macro_indicators(start_date="2007-01-01", end_date="2024-12-31"):
    """
    Downloads macroeconomic indicators from FRED.
    SP500 is fetched separately via yfinance for better historical coverage.
    Returns a DataFrame indexed by date.
    """
    fred = get_fred_client()

    indicators = {
        "fed_funds_rate": "FEDFUNDS",
        "unemployment_rate": "UNRATE",
        "cpi_inflation": "CPIAUCSL",
        "gdp_growth": "A191RL1Q225SBEA",
    }

    print("Downloading macroeconomic indicators from FRED...")
    frames = {}
    for name, series_id in indicators.items():
        for attempt in range(3):
            try:
                print(f"  -> {name} ({series_id})")
                frames[name] = fred.get_series(series_id, start_date, end_date)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"     Retrying ({attempt + 1}/3)...")
                    import time
                    time.sleep(3)
                else:
                    print(f"     Failed after 3 attempts: {e}")
                    raise

    print("  -> sp500 (yfinance)")
    import yfinance as yf
    sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
    sp500_monthly = sp500["Close"].squeeze().resample("MS").last()
    frames["sp500"] = sp500_monthly

    macro_df = pd.DataFrame(frames)
    macro_df.index.name = "date"
    macro_df = macro_df.resample("MS").last()
    macro_df = macro_df.ffill()
    macro_df = macro_df.bfill()
    macro_df = macro_df.dropna()

    print(f"Macro data: {macro_df.shape[0]} rows, {macro_df.shape[1]} columns")
    return macro_df



def validate_dataframe(df, name):
    """
    Runs basic validation checks on a DataFrame.
    """
    print(f"\nValidating {name}...")
    print(f"  Shape: {df.shape}")
    print(f"  Nulls:\n{df.isnull().sum()}")
    print(f"  Dtypes:\n{df.dtypes}")
    return df


def load_lending_club(filepath="data/raw/accepted_2007_to_2018Q4.csv", nrows=500000):
    """
    Loads Lending Club loan data.
    Uses nrows to limit memory usage during development.
    """
    print(f"Loading Lending Club data ({nrows:,} rows)...")

    usecols = [
        "loan_amnt", "term", "int_rate", "installment", "grade",
        "sub_grade", "emp_length", "home_ownership", "annual_inc",
        "verification_status", "issue_d", "loan_status", "purpose",
        "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
        "mort_acc", "pub_rec_bankruptcies"
    ]

    df = pd.read_csv(filepath, usecols=usecols, nrows=nrows, low_memory=False)
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def create_target_variable(df):
    """
    Creates binary target variable from loan_status.
    1 = default, 0 = fully paid.
    Drops rows with ambiguous loan status.
    """
    keep_status = ["Fully Paid", "Charged Off", "Default"]
    df = df[df["loan_status"].isin(keep_status)].copy()

    df["default"] = (df["loan_status"].isin(["Charged Off", "Default"])).astype(int)
    df = df.drop(columns=["loan_status"])

    default_rate = df["default"].mean() * 100
    print(f"Target created — Default rate: {default_rate:.1f}%")
    print(f"Rows after filtering: {df.shape[0]:,}")
    return df





if __name__ == "__main__":
    # FRED macro indicators
    macro = fetch_macro_indicators()
    validate_dataframe(macro, "macro_indicators")
    os.makedirs("data/processed", exist_ok=True)
    macro.to_csv("data/processed/macro_indicators.csv")
    print("Saved to data/processed/macro_indicators.csv")

    # Lending Club loans
    loans = load_lending_club()
    loans = create_target_variable(loans)
    validate_dataframe(loans, "lending_club")
    loans.to_csv("data/processed/loans_clean.csv", index=False)
    print("Saved to data/processed/loans_clean.csv") 