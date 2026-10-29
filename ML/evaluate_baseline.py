
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------
ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR / "Data" / "Processed"
MODEL_DIR = ML_DIR / "Saved Models"

DATA_FILE = DATA_DIR / "train_CA_1.parquet"
MODEL_FILE = MODEL_DIR / "lightgbm_ca1.txt"
META_FILE = MODEL_DIR / "lightgbm_ca1_features.json"

MAX_ITEMS = 200
LOOKBACK_DAYS = 365
VALIDATION_DAYS = 28
BATCH_SIZE = 250_000

# --------------------------------------------------
# 2. Load saved feature information
# --------------------------------------------------
with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

features = metadata["features"]
categorical_features = metadata["categorical_features"]

parquet_file = pq.ParquetFile(DATA_FILE)
available = parquet_file.schema.names

required = list(dict.fromkeys(
    ["d", "sales", "item_id"] + features
))

missing = [col for col in required if col not in available]
if missing:
    raise ValueError(f"Missing columns in Parquet file: {missing}")

# Match the training script's date range
max_day = int(metadata["max_day"])
first_day = max(1, max_day - LOOKBACK_DAYS + 1)
validation_start = max_day - VALIDATION_DAYS + 1

print("Loading validation data in batches...")

parts = []

for batch in parquet_file.iter_batches(
    batch_size=BATCH_SIZE,
    columns=required
):
    chunk = batch.to_pandas()

    chunk["d"] = pd.to_numeric(chunk["d"], errors="coerce")
    chunk["item_id"] = pd.to_numeric(
        chunk["item_id"], errors="coerce"
    )

    mask = (
        chunk["item_id"].lt(MAX_ITEMS)
        & chunk["d"].between(first_day, max_day)
    )

    chunk = chunk.loc[mask]

    if not chunk.empty:
        parts.append(chunk)

if not parts:
    raise ValueError(
        "No matching rows found. Check item_id encoding and date range."
    )

data = pd.concat(parts, ignore_index=True)
del parts

data = data.dropna(subset=["d", "sales", "item_id"]).copy()
data["d"] = data["d"].astype("int32")

# --------------------------------------------------
# 3. Select the same validation period
# --------------------------------------------------
valid = data.loc[data["d"] >= validation_start].copy()

if valid.empty:
    raise ValueError("No validation rows found.")

# --------------------------------------------------
# 4. LightGBM predictions
# --------------------------------------------------
X_valid = valid[features].copy()

for col in categorical_features:
    if col in X_valid.columns:
        X_valid[col] = X_valid[col].astype("category")

model = lgb.Booster(model_file=str(MODEL_FILE))

predictions = np.maximum(model.predict(X_valid), 0)
actual = valid["sales"].to_numpy(dtype=float)

# --------------------------------------------------
# 5. Seven-day seasonal baseline
# --------------------------------------------------
# Find actual sales from exactly seven days earlier
history = data[["item_id", "d", "sales"]].copy()
history["d"] += 7
history = history.rename(columns={"sales": "baseline_prediction"})

comparison = valid.merge(
    history,
    on=["item_id", "d"],
    how="left",
    validate="many_to_one"
)

baseline_mask = comparison["baseline_prediction"].notna()

if baseline_mask.sum() == 0:
    raise ValueError(
        "No seven-day baseline matches found. "
        "Check the item IDs and available historical dates."
    )

# Compare both methods on exactly the same rows
actual_common = comparison.loc[baseline_mask, "sales"].to_numpy(
    dtype=float
)
baseline = comparison.loc[
    baseline_mask, "baseline_prediction"
].to_numpy(dtype=float)

# Align LightGBM predictions to those same rows
lightgbm_predictions = predictions[baseline_mask.to_numpy()]

# --------------------------------------------------
# 6. Metrics
# --------------------------------------------------
def report_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{name}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return mae, rmse


print("\n--- EVALUATION RESULTS ---")
print("Validation rows available:", len(valid))
print("Rows compared fairly:", int(baseline_mask.sum()))

model_mae, model_rmse = report_metrics(
    "LightGBM",
    actual_common,
    lightgbm_predictions
)

base_mae, base_rmse = report_metrics(
    "Seven-day baseline",
    actual_common,
    baseline
)

print("\n--- COMPARISON ---")
if base_mae > 0:
    improvement = (base_mae - model_mae) / base_mae * 100
    print(f"MAE improvement: {improvement:.2f}%")
else:
    print("MAE improvement: baseline MAE is zero; percentage is undefined.")

if model_mae < base_mae:
    print("LightGBM has lower MAE on this validation subset.")
elif model_mae > base_mae:
    print("The seven-day baseline has lower MAE on this validation subset.")
else:
    print("Both methods have the same MAE on this validation subset.")

print("\nEvaluation completed.")
