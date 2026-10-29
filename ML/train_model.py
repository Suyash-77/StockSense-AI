
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================

ML_DIR = Path(__file__).resolve().parent

DATA_DIR = ML_DIR / "Data" / "Processed"
MODEL_DIR = ML_DIR / "Saved Models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH = DATA_DIR / "train_CA_1.parquet"

MODEL_PATH = MODEL_DIR / "lightgbm_ca1.txt"
METADATA_PATH = MODEL_DIR / "lightgbm_ca1_features.json"

# Training settings
MAX_ITEMS = 200
LOOKBACK_DAYS = 365
VALIDATION_DAYS = 28
BATCH_SIZE = 250_000


# ==================================================
# 2. CHECK DATASET
# ==================================================

if not FILE_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{FILE_PATH}"
    )

print("\n========== STOCKSENSE AI ==========")
print("\nInspecting Parquet dataset...")

parquet_file = pq.ParquetFile(FILE_PATH)
available = parquet_file.schema.names

if "sales" not in available or "d" not in available:
    raise ValueError(
        "Dataset must contain 'sales' and 'd' columns."
    )

candidate_features = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "sell_price",
    "weekday",
    "month",
    "year",
    "has_event",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap",
    "lag_28",
    "lag_35",
    "lag_42",
    "roll_mean_7_lag28",
    "roll_mean_28_lag28",
]

feature_columns = [
    col for col in candidate_features
    if col in available
]

required_features = [
    "item_id",
    "lag_28",
]

missing_required = [
    col for col in required_features
    if col not in feature_columns
]

if missing_required:
    raise ValueError(
        f"Required features are missing: {missing_required}"
    )

read_columns = list(
    dict.fromkeys(["d", "sales"] + feature_columns)
)

print("Features:", feature_columns)


# ==================================================
# 3. FIND THE LATEST DAY
# ==================================================

print("\nFinding the latest available day...")

max_day = None

for batch in parquet_file.iter_batches(
    batch_size=BATCH_SIZE,
    columns=["d"],
):
    day_values = pd.to_numeric(
        batch.column("d").to_pandas(),
        errors="coerce",
    )

    day_max = day_values.max()

    if pd.notna(day_max):
        if max_day is None:
            max_day = day_max
        else:
            max_day = max(max_day, day_max)

if max_day is None:
    raise ValueError("Could not determine the latest day.")

max_day = int(max_day)

first_day = max(
    1,
    max_day - LOOKBACK_DAYS + 1,
)

validation_start = max_day - VALIDATION_DAYS + 1

print("Latest day:", max_day)
print("First day used:", first_day)
print("Validation starts at:", validation_start)


# ==================================================
# 4. LOAD SELECTED PRODUCTS AND DATES
# ==================================================

print("\nReading dataset in batches...")

parts = []

for batch in parquet_file.iter_batches(
    batch_size=BATCH_SIZE,
    columns=read_columns,
):
    chunk = batch.to_pandas()

    chunk["d"] = pd.to_numeric(
        chunk["d"],
        errors="coerce",
    )

    # Keep a manageable subset of encoded item IDs.
    mask = (
        (chunk["item_id"] < MAX_ITEMS)
        & (chunk["d"] >= first_day)
        & (chunk["d"] <= max_day)
    )

    selected = chunk.loc[mask].copy()

    if not selected.empty:
        parts.append(selected)

if not parts:
    raise ValueError(
        "No rows selected. Check item_id encoding and date values."
    )

data = pd.concat(parts, ignore_index=True)

del parts

data["d"] = pd.to_numeric(
    data["d"],
    errors="coerce",
)

data["sales"] = pd.to_numeric(
    data["sales"],
    errors="coerce",
)

data = data.dropna(subset=["d", "sales"]).copy()

data["d"] = data["d"].astype("int32")

# Sales cannot be negative.
data = data[data["sales"] >= 0].copy()

if data.empty:
    raise ValueError("No valid training rows remain.")

print("Selected rows:", len(data))
print("Selected products:", data["item_id"].nunique())


# ==================================================
# 5. PREPARE FEATURES AND TARGET
# ==================================================

print("\nPreparing features...")

X = data[feature_columns].copy()

y = data["sales"].astype("float32")

days = data["d"]

# Treat encoded identifiers and event fields as categories.
categorical_candidates = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "weekday",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]

categorical_features = [
    col for col in categorical_candidates
    if col in X.columns
]

for col in categorical_features:
    X[col] = X[col].astype("category")

print("Categorical features:", categorical_features)


# ==================================================
# 6. CHRONOLOGICAL TRAIN-VALIDATION SPLIT
# ==================================================

print("\nSplitting training and validation data...")

train_mask = days < validation_start
valid_mask = days >= validation_start

X_train = X.loc[train_mask].copy()
y_train = y.loc[train_mask].copy()

X_valid = X.loc[valid_mask].copy()
y_valid = y.loc[valid_mask].copy()

if X_train.empty or X_valid.empty:
    raise ValueError(
        "Training or validation data is empty."
    )

print("Training rows:", len(X_train))
print("Validation rows:", len(X_valid))
print("Training days:", days.loc[train_mask].nunique())
print("Validation days:", days.loc[valid_mask].nunique())


# ==================================================
# 7. TRAIN LIGHTGBM MODEL
# ==================================================

print("\nTraining LightGBM model...")

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    verbosity=-1,
    n_jobs=4,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
    categorical_feature=categorical_features,
    eval_set=[(X_valid, y_valid)],
    callbacks=[
        lgb.early_stopping(30, verbose=True),
        lgb.log_evaluation(25),
    ],
)

print("\nTraining completed.")
print("Best iteration:", model.best_iteration_)


# ==================================================
# 8. EVALUATE LIGHTGBM MODEL
# ==================================================

print("\n========== MODEL EVALUATION ==========")

predictions = model.predict(X_valid)

# Demand cannot be negative.
predictions = np.maximum(predictions, 0)

mae = mean_absolute_error(
    y_valid,
    predictions,
)

rmse = np.sqrt(
    mean_squared_error(y_valid, predictions)
)

total_actual_sales = float(
    np.abs(y_valid.to_numpy()).sum()
)

if total_actual_sales > 0:
    wape = (
        np.abs(
            y_valid.to_numpy() - predictions
        ).sum()
        / total_actual_sales
    ) * 100
else:
    wape = None

print("\n--- LightGBM Results ---")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

if wape is not None:
    print(f"WAPE: {wape:.2f}%")
else:
    print("WAPE: Cannot calculate because actual sales total zero.")

print("Validation rows:", len(y_valid))
print("Validation days:", VALIDATION_DAYS)


# ==================================================
# 9. BASELINE COMPARISON
# ==================================================

print("\n--- Baseline: Sales from 28 Days Earlier ---")

# Use the same validation rows for both methods.
baseline_predictions = pd.to_numeric(
    X_valid["lag_28"],
    errors="coerce",
)

# Exclude rows where the baseline feature is missing.
baseline_mask = baseline_predictions.notna()

if baseline_mask.any():

    baseline_actual = y_valid.loc[baseline_mask]

    baseline_predictions = np.maximum(
        baseline_predictions.loc[baseline_mask].to_numpy(),
        0,
    )

    baseline_mae = mean_absolute_error(
        baseline_actual,
        baseline_predictions,
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            baseline_actual,
            baseline_predictions,
        )
    )

    print(f"Baseline MAE:  {baseline_mae:.4f}")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")

    if baseline_mae > 0:
        improvement = (
            (baseline_mae - mae) / baseline_mae
        ) * 100

        print(
            f"MAE improvement over baseline: "
            f"{improvement:.2f}%"
        )
    else:
        print(
            "Baseline MAE is zero; "
            "improvement percentage is undefined."
        )

else:
    print("Baseline could not be evaluated.")

print("\n======================================")


# ==================================================
# 10. SAVE MODEL AND FEATURE METADATA
# ==================================================

print("\nSaving model...")

model.booster_.save_model(str(MODEL_PATH))

metadata = {
    "features": feature_columns,
    "categorical_features": categorical_features,
    "validation_days": VALIDATION_DAYS,
    "max_day": max_day,
    "first_day": first_day,
    "max_items": MAX_ITEMS,
    "mae": float(mae),
    "rmse": float(rmse),
    "wape": float(wape) if wape is not None else None,
    "best_iteration": int(model.best_iteration_),
}

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        metadata,
        file,
        indent=2,
    )

print("Model saved to:", MODEL_PATH)
print("Metadata saved to:", METADATA_PATH)


# ==================================================
# 11. FINAL STATUS
# ==================================================

print("\n========== TRAINING SUMMARY ==========")
print("Model: LightGBM")
print("Products selected:", data["item_id"].nunique())
print("Training rows:", len(X_train))
print("Validation rows:", len(X_valid))
print(f"Final MAE: {mae:.4f}")
print(f"Final RMSE: {rmse:.4f}")

if wape is not None:
    print(f"Final WAPE: {wape:.2f}%")

print("\nStockSense AI training and evaluation completed!")
