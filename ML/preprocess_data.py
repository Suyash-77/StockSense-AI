
from pathlib import Path
import pandas as pd

# Project paths
ML_DIR = Path(__file__).resolve().parent
RAW_DIR = ML_DIR / "Data" / "Raw"
PROCESSED_DIR = ML_DIR / "Data" / "Processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load a small subset of the sales dataset
print("Loading sample sales data...")

sales = pd.read_csv(
    RAW_DIR / "sales_train_validation.csv",
    nrows=50
)

# Identify daily sales columns
date_columns = [
    col for col in sales.columns
    if col.startswith("d_")
]

# Keep identifiers and daily sales
id_columns = [
    "id", "item_id", "dept_id",
    "cat_id", "store_id", "state_id"
]

sales = sales[id_columns + date_columns]

# 2. Convert wide sales data into long format
print("Reshaping sales data...")

data = sales.melt(
    id_vars=id_columns,
    value_vars=date_columns,
    var_name="d",
    value_name="sales"
)

data["sales"] = data["sales"].astype("int16")

# 3. Add calendar information
print("Joining calendar data...")

calendar = pd.read_csv(RAW_DIR / "calendar.csv")

calendar_columns = [
    "d", "wm_yr_wk", "weekday",
    "month", "year", "event_name_1",
    "event_type_1", "snap_CA",
    "snap_TX", "snap_WI"
]

calendar_columns = [
    col for col in calendar_columns
    if col in calendar.columns
]

data = data.merge(
    calendar[calendar_columns],
    on="d",
    how="left",
    validate="many_to_one"
)

# 4. Add historical sales features
print("Creating lag and rolling features...")

data = data.sort_values(["id", "d"]).reset_index(drop=True)

grouped_sales = data.groupby("id", sort=False)["sales"]

data["lag_7"] = grouped_sales.shift(7)
data["lag_28"] = grouped_sales.shift(28)

data["rolling_mean_7"] = (
    data.groupby("id", sort=False)["sales"]
    .transform(lambda s: s.shift(1).rolling(7).mean())
)

# 5. Encode calendar categories
for col in ["weekday", "event_name_1", "event_type_1"]:
    if col in data.columns:
        data[col] = data[col].fillna("None").astype("category").cat.codes

# 6. Remove rows without enough historical information
data = data.dropna(
    subset=["lag_7", "lag_28", "rolling_mean_7"]
).copy()

# Keep the target and model features
feature_columns = [
    "id", "item_id", "store_id", "d",
    "sales", "lag_7", "lag_28",
    "rolling_mean_7", "month", "year"
]

feature_columns = [
    col for col in feature_columns
    if col in data.columns
]

data = data[feature_columns]

# 7. Save the processed sample
output_path = PROCESSED_DIR / "training_sample.csv"
data.to_csv(output_path, index=False)

print("\nPreprocessing completed!")
print("Rows:", len(data))
print("Columns:", data.columns.tolist())
print("Saved to:", output_path)
print(data.head().to_string(index=False))
