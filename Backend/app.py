
from pathlib import Path
import json

import lightgbm as lgb
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_DIR / "ML" / "Data" / "Raw"
PROCESSED_DIR = PROJECT_DIR / "ML" / "Data" / "Processed"
MODEL_DIR = PROJECT_DIR / "ML" / "Saved Models"

MODEL_PATH = MODEL_DIR / "lightgbm_ca1.txt"
METADATA_PATH = MODEL_DIR / "lightgbm_ca1_features.json"
MAPPINGS_PATH = PROCESSED_DIR / "mappings.json"
SALES_PATH = PROCESSED_DIR / "train_CA_1.parquet"
CALENDAR_PATH = RAW_DIR / "calendar.csv"
PRICES_PATH = RAW_DIR / "sell_prices.csv"


# ==================================================
# 2. LOAD MODEL AND DATA
# ==================================================

model = None
metadata = None
mappings = None
sales = None
calendar = None
prices = None
product_mapping = {}
startup_error = None
store_code = None

try:
    required_files = [
        MODEL_PATH,
        METADATA_PATH,
        MAPPINGS_PATH,
        SALES_PATH,
        CALENDAR_PATH,
        PRICES_PATH,
    ]

    missing_files = [
        str(path) for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required files: " + ", ".join(missing_files)
        )

    # Load trained LightGBM model
    model = lgb.Booster(model_file=str(MODEL_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    with open(MAPPINGS_PATH, "r", encoding="utf-8") as file:
        mappings = json.load(file)

    # Load sales history and product identifiers
    sales = pd.read_parquet(
        SALES_PATH,
        columns=[
            "item_id",
            "dept_id",
            "cat_id",
            "state_id",
            "d",
            "sales",
        ],
    )

    sales["item_id"] = sales["item_id"].astype(int)
    sales["d"] = sales["d"].astype(int)

    # Load calendar
    calendar = pd.read_csv(CALENDAR_PATH)

    calendar["day_number"] = (
        calendar["d"]
        .str.replace("d_", "", regex=False)
        .astype(int)
    )

    # Load prices
    prices = pd.read_csv(PRICES_PATH)

    # Model was trained for CA_1
    store_code = next(
        (
            int(code)
            for code, name in mappings["store_id"].items()
            if name == "CA_1"
        ),
        None,
    )

    if store_code is None:
        raise ValueError("CA_1 was not found in mappings.json.")

    # Only IDs 0 through 199 were selected by the training script
    product_mapping = {
        int(code): product
        for code, product in mappings["item_id"].items()
        if int(code) < 200
    }

    # Confirm the model's expected feature count
    if model.num_feature() != len(metadata["features"]):
        raise ValueError(
            "Model feature count does not match saved metadata."
        )

    print("StockSense AI model loaded successfully.")
    print("Model feature count:", model.num_feature())
    print("Supported products:", len(product_mapping))

except Exception as exc:
    startup_error = str(exc)
    print("StockSense AI startup error:", startup_error)


# ==================================================
# 3. HELPER FUNCTIONS
# ==================================================

def get_product_id(product):
    """Find an item ID using its product code or encoded ID."""

    # Product code, for example FOODS_1_001
    for item_id, product_code in product_mapping.items():
        if product_code == product:
            return item_id

    # Encoded ID, for example 0
    try:
        item_id = int(product)

        if item_id in product_mapping:
            return item_id

    except (ValueError, TypeError):
        pass

    return None


def encode_event(mapping_name, value):
    """Convert a calendar event into its training-time category code."""

    event_mapping = mappings[mapping_name]

    if pd.isna(value) or str(value).strip() == "":
        value = "none"

    for code, label in event_mapping.items():
        if str(label) == str(value):
            return int(code)

    # Unknown or missing event: use the encoded 'none' category.
    for code, label in event_mapping.items():
        if str(label).lower() == "none":
            return int(code)

    return 0


def get_item_information(item_id):
    """Get the product's encoded department, category and state."""

    rows = sales.loc[sales["item_id"] == item_id]

    if rows.empty:
        raise ValueError(
            f"No historical sales found for item ID {item_id}."
        )

    item = rows.iloc[0]

    return {
        "dept_id": int(item["dept_id"]),
        "cat_id": int(item["cat_id"]),
        "state_id": int(item["state_id"]),
    }


def get_historical_sales(item_id):
    """Return the product's sales indexed by day number."""

    rows = sales.loc[sales["item_id"] == item_id]

    if rows.empty:
        raise ValueError(
            f"No historical sales found for item ID {item_id}."
        )

    return rows.set_index("d")["sales"]


def get_sales_value(sales_by_day, day_number):
    """Retrieve a historical sales value for a specified day."""

    if day_number not in sales_by_day.index:
        raise ValueError(
            f"Historical sales are missing for day d_{day_number}."
        )

    value = sales_by_day.loc[day_number]

    # A duplicate day would make the value a Series.
    if isinstance(value, pd.Series):
        value = value.iloc[0]

    return float(value)


def get_product_price(product_code, target_week):
    """
    Find the product's price for the target week.

    If no price exists for that week, use its latest available
    price from an earlier week and report that fallback.
    """

    product_prices = prices.loc[
        (prices["store_id"] == "CA_1")
        & (prices["item_id"] == product_code)
    ]

    if product_prices.empty:
        raise ValueError(
            f"No price history found for product {product_code}."
        )

    current_week = product_prices.loc[
        product_prices["wm_yr_wk"] == target_week
    ]

    if not current_week.empty:
        return float(current_week.iloc[0]["sell_price"]), False

    earlier_prices = product_prices.loc[
        product_prices["wm_yr_wk"] <= target_week
    ].sort_values("wm_yr_wk")

    if earlier_prices.empty:
        raise ValueError(
            f"No current or earlier price available for {product_code}."
        )

    return float(earlier_prices.iloc[-1]["sell_price"]), True


def make_features(item_id, target_day):
    """Build one model input row for a future calendar day."""

    product_code = product_mapping[item_id]

    # Get the calendar row for this day
    day_rows = calendar.loc[
        calendar["day_number"] == target_day
    ]

    if day_rows.empty:
        raise ValueError(
            f"Calendar information for d_{target_day} is unavailable."
        )

    day = day_rows.iloc[0]

    # Historical sales for the selected product
    sales_by_day = get_historical_sales(item_id)

    # The model uses sales from 28, 35 and 42 days earlier.
    lag_28 = get_sales_value(sales_by_day, target_day - 28)
    lag_35 = get_sales_value(sales_by_day, target_day - 35)
    lag_42 = get_sales_value(sales_by_day, target_day - 42)

    # Rolling averages calculated from the lagged sales history.
    mean_7 = sum(
        get_sales_value(sales_by_day, target_day - 28 - offset)
        for offset in range(7)
    ) / 7

    mean_28 = sum(
        get_sales_value(sales_by_day, target_day - 28 - offset)
        for offset in range(28)
    ) / 28

    # Price for this product, store and calendar week
    sell_price, price_fallback = get_product_price(
        product_code,
        day["wm_yr_wk"],
    )

    item_info = get_item_information(item_id)

    # Detect whether either event field is populated.
    has_event = int(
        not pd.isna(day["event_name_1"])
        or not pd.isna(day["event_name_2"])
    )

    # Feature names must match the saved model metadata.
    all_features = {
        "item_id": item_id,
        "dept_id": item_info["dept_id"],
        "cat_id": item_info["cat_id"],
        "store_id": store_code,
        "state_id": item_info["state_id"],
        "sell_price": sell_price,
        "month": int(day["month"]),
        "year": int(day["year"]),
        "has_event": has_event,
        "event_name_1": encode_event(
            "event_name_1", day["event_name_1"]
        ),
        "event_type_1": encode_event(
            "event_type_1", day["event_type_1"]
        ),
        "event_name_2": encode_event(
            "event_name_2", day["event_name_2"]
        ),
        "event_type_2": encode_event(
            "event_type_2", day["event_type_2"]
        ),
        "snap": int(day["snap_CA"]),
        "lag_28": lag_28,
        "lag_35": lag_35,
        "lag_42": lag_42,
        "roll_mean_7_lag28": mean_7,
        "roll_mean_28_lag28": mean_28,
    }

    # Preserve the exact feature order used during training.
    row = pd.DataFrame(
        [[all_features[name] for name in metadata["features"]]],
        columns=metadata["features"],
    )

    # Restore categorical dtypes used by the training pipeline.
    for column in metadata.get("categorical_features", []):
        if column in row.columns:
            row[column] = row[column].astype("category")

    return row, price_fallback


# ==================================================
# 4. API ENDPOINTS
# ==================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "project": "StockSense AI",
        "message": "Inventory forecasting backend is running",
    })


@app.route("/api/health", methods=["GET"])
def health():
    healthy = startup_error is None and model is not None

    return jsonify({
        "status": "healthy" if healthy else "error",
        "model_loaded": model is not None,
        "error": startup_error,
    }), (200 if healthy else 500)


@app.route("/api/model-info", methods=["GET"])
def model_info():
    if model is None:
        return jsonify({
            "error": startup_error or "Model is not loaded."
        }), 500

    return jsonify({
        "model": "LightGBM",
        "store": "CA_1",
        "feature_count": len(metadata["features"]),
        "features": metadata["features"],
    })


@app.route("/api/products", methods=["GET"])
def products():
    if model is None:
        return jsonify({
            "error": startup_error or "Model is not loaded."
        }), 500

    return jsonify({
        "store": "CA_1",
        "count": len(product_mapping),
        "products": [
            {
                "item_id": item_id,
                "product_code": product_code,
            }
            for item_id, product_code
            in sorted(product_mapping.items())
        ],
    })


@app.route("/api/forecast/<product>", methods=["GET"])
def forecast(product):
    if model is None:
        return jsonify({
            "error": startup_error or "Model is not loaded."
        }), 500

    item_id = get_product_id(product)

    if item_id is None:
        return jsonify({
            "error": "Product not supported by this model.",
            "message": (
                "Use a product code or item ID returned by /api/products."
            ),
        }), 404

    try:
        # Forecast after the final day with recorded sales.
        last_sales_day = int(sales["d"].max())
        forecast_days = range(
            last_sales_day + 1,
            last_sales_day + 8,
        )

        results = []
        used_price_fallback = False

        for day_number in forecast_days:
            features, price_fallback = make_features(
                item_id,
                day_number,
            )

            prediction = float(model.predict(features)[0])
            prediction = max(0.0, prediction)

            date_rows = calendar.loc[
                calendar["day_number"] == day_number, "date"
            ]

            if date_rows.empty:
                raise ValueError(
                    f"Date for d_{day_number} is unavailable."
                )

            results.append({
                "day": f"d_{day_number}",
                "date": str(date_rows.iloc[0]),
                "predicted_sales": round(prediction, 2),
            })

            used_price_fallback = (
                used_price_fallback or price_fallback
            )

        return jsonify({
            "product_code": product_mapping[item_id],
            "item_id": item_id,
            "store": "CA_1",
            "forecast_horizon_days": 7,
            "forecast_type": "historical M5 dataset forecast",
            "price_fallback_used": used_price_fallback,
            "forecasts": results,
        })

    except (ValueError, KeyError, IndexError) as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception as exc:
        app.logger.exception("Forecast failed")
        return jsonify({
            "error": "Unable to generate forecast.",
            "details": str(exc),
        }), 500


# ==================================================
# 5. START FLASK SERVER
# ==================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
