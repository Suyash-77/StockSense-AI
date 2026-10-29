
from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "Data" / "Raw"

# Inspect the main sales dataset without loading the entire file
sales_file = RAW_DIR / "sales_train_validation.csv"

sales = pd.read_csv(sales_file, nrows=5)

print("\n--- SALES DATA ---")
print("Shape of 5-row sample:", sales.shape)
print("\nFirst columns:")
print(sales.columns[:10].tolist())
print("\nLast columns:")
print(sales.columns[-10:].tolist())
print("\nSample records:")
print(sales.iloc[:, :10].to_string(index=False))

# Inspect calendar data
calendar = pd.read_csv(RAW_DIR / "calendar.csv")

print("\n--- CALENDAR DATA ---")
print("Shape:", calendar.shape)
print("Missing values:")
print(calendar.isnull().sum().to_string())
print("\nFirst records:")
print(calendar.head(3).to_string(index=False))

# Inspect price data using only a small sample
prices = pd.read_csv(RAW_DIR / "sell_prices.csv", nrows=5)

print("\n--- SELL PRICES ---")
print("Columns:", prices.columns.tolist())
print(prices.to_string(index=False))

print("\nData checks completed.")
