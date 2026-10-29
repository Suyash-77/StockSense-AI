
from pathlib import Path
import json

import lightgbm as lgb
import pandas as pd


# ==================================================
# 1. PATHS
# ==================================================

ML_DIR = Path(__file__).resolve().parent

MODEL_PATH = ML_DIR / "Saved Models" / "lightgbm_ca1.txt"
DATA_PATH = ML_DIR / "Data" / "Processed" / "train_CA_1.parquet"
METADATA_PATH = ML_DIR / "Saved Models" / "lightgbm_ca1_features.json"


# ==================================================
# 2. CHECK FILES
# ==================================================

print("\n========== STOCKSENSE AI ==========")

for path in [MODEL_PATH, DATA_PATH, METADATA_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")


# ==================================================
# 3. LOAD MODEL, METADATA AND DATA
# ==================================================

print("\nLoading model...")

model = lgb.Booster(model_file=str(MODEL_PATH))

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

print("Model loaded successfully!")

print("\nLoading processed dataset...")

df = pd.read_parquet(DATA_PATH)

print("Dataset shape:", df.shape)


# ==================================================
# 4. VERIFY FEATURES
# ==================================================

features = metadata["features"]
categorical_features = metadata["categorical_features"]

if features != model.feature_name():
    raise ValueError(
        "Feature order in the metadata does not match the model."
    )

missing = [col for col in features if col not in df.columns]

if missing:
    raise ValueError(f"Missing features: {missing}")

print("\nModel features verified.")


# ==================================================
# 5. LOAD SAVED CATEGORY MAPPINGS
# ==================================================

saved_categories = model.pandas_categorical

if saved_categories is None:
    raise ValueError(
        "The model has no saved pandas category metadata."
    )

if len(saved_categories) != len(categorical_features):
    raise ValueError(
        "The number of saved category mappings does not match "
        "the categorical features.\n"
        f"Saved mappings: {len(saved_categories)}\n"
        f"Categorical features: {len(categorical_features)}\n"
        "Check the original training process."
    )

print("\nSaved category mappings verified.")

for col, categories in zip(
    categorical_features, saved_categories
):
    print(f"{col}: {len(categories)} categories")


# ==================================================
# 6. FIND A ROW COMPATIBLE WITH TRAINING CATEGORIES
# ==================================================

# The training script selected item IDs below 200.
# Also require all categorical values to exist in the
# model's saved category mappings.

valid_mask = df["item_id"] < 200

for col, categories in zip(
    categorical_features, saved_categories
):
    valid_mask &= df[col].isin(categories)

compatible_rows = df.loc[valid_mask, features]

if compatible_rows.empty:
    raise ValueError(
        "No compatible rows found in the dataset.\n"
        "The training and processed datasets may use different "
        "encodings. Inspect preprocess_data.py and train_model.py."
    )

# Select the last compatible row for a test prediction.
X = compatible_rows.tail(1).copy()

print("\nCompatible test row:")
print(X.to_string(index=False))


# ==================================================
# 7. APPLY THE MODEL'S CATEGORY MAPPINGS
# ==================================================

# Keep categorical columns in the same order used during training.
for col, categories in zip(
    categorical_features, saved_categories
):
    X[col] = pd.Categorical(
        X[col],
        categories=categories
    )

# Check that no categorical value became missing.
missing_categories = [
    col for col in categorical_features
    if X[col].isna().any()
]

if missing_categories:
    raise ValueError(
        "Category conversion produced missing values in: "
        f"{missing_categories}"
    )


# ==================================================
# 8. GENERATE PREDICTION
# ==================================================

print("\n========== PREDICTION ==========")

prediction = model.predict(X)

predicted_demand = max(0.0, float(prediction[0]))

print("Prediction successful!")
print("Predicted demand:", round(predicted_demand, 2), "units")

print("\nTest item ID:", int(X["item_id"].iloc[0]))

print(
    "\nNote: This verifies compatibility with the saved "
    "category mappings. It does not establish forecast accuracy."
)


# ==================================================
# 9. FINISH
# ==================================================

print("\n========== TEST COMPLETED ==========")
