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
    Returns a DataFrame indexed by date.
    Includes retry logic for temporary server errors.
    """
    fred = get_fred_client()

    indicators = {
        "fed_funds_rate": "FEDFUNDS",
        "unemployment_rate": "UNRATE",
        "cpi_inflation": "CPIAUCSL",
        "gdp_growth": "A191RL1Q225SBEA",
        "sp500": "SP500",
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

    macro_df = pd.DataFrame(frames)
    macro_df.index.name = "date"
    macro_df = macro_df.resample("MS").last()
    macro_df = macro_df.ffill()
    macro_df = macro_df.dropna(subset=["sp500"])

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


if __name__ == "__main__":
    macro = fetch_macro_indicators()
    validate_dataframe(macro, "macro_indicators")

    os.makedirs("data/processed", exist_ok=True)
    macro.to_csv("data/processed/macro_indicators.csv")
    print("\nSaved to data/processed/macro_indicators.csv")