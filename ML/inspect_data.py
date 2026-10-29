
from pathlib import Path
import pandas as pd

# Locate original dataset files
RAW_DIR = Path(__file__).resolve().parent / "Data" / "Raw"

files = [
    "calendar.csv",
    "sales_train_validation.csv",
    "sales_train_evaluation.csv",
    "sell_prices.csv",
    "sample_submission.csv",
]

for filename in files:
    path = RAW_DIR / filename

    print("\n" + "=" * 65)
    print(f"FILE: {filename}")
    print("=" * 65)

    if not path.exists():
        print("File not found:", path)
        continue

    print(f"File size: {path.stat().st_size / (1024**2):.2f} MB")

    # Read only five rows to limit memory usage
    df = pd.read_csv(path, nrows=5)

    print("Number of columns:", len(df.columns))
    print("Column names:", df.columns.tolist())
    print("\nSample rows:")
    print(df.head().to_string(index=False))
    print("\nData types:")
    print(df.dtypes.to_string())

print("\nInspection complete.")
