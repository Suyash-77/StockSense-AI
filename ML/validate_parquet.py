
from pathlib import Path
import pandas as pd

ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR / "Data" / "Processed"

file_path = DATA_DIR / "train_CA_1.parquet"

df = pd.read_parquet(file_path)

print("File loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().to_string())

print("\nFirst five rows:")
print(df.head().to_string(index=False))

print("\nData types:")
print(df.dtypes.to_string())
