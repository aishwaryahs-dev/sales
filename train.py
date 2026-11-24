"""Training pipeline: data -> validation -> features -> train -> acceptance gate -> save model.
If the new model does not beat the baseline, this script fails, and so does the CI pipeline."""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from features import make_features

# Change this number to release a new model version (for example "1.1").
MODEL_VERSION = "1.0"

SEED = 42
N_STORES = 5


def generate_sales_data(seed: int = SEED) -> pd.DataFrame:
    """Two years of daily sales for 5 stores (our batch data source)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=730, freq="D")
    rows = []
    for store in range(1, N_STORES + 1):
        base = 80 + store * 20
        for d in dates:
            weekend = 1.3 if d.dayofweek >= 5 else 1.0
            season = 1 + 0.15 * np.sin(2 * np.pi * d.dayofyear / 365)
            rows.append({"date": d, "store_id": store,
                         "sales": round(base * weekend * season + rng.normal(0, 8))})
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame) -> dict:
    """Data quality checks that must pass before training."""
    return {
        "schema": list(df.columns) == ["date", "store_id", "sales"],
        "no_missing_sales": bool(df["sales"].notna().all()),
        "sales_in_range": bool(df["sales"].between(0, 1000).all()),
        "no_duplicates": not df.duplicated(["date", "store_id"]).any(),
    }


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Signals from past days only, so there is no leakage."""
    df = df.sort_values(["store_id", "date"]).copy()
    by_store = df.groupby("store_id")["sales"]
    df["last_week_sales"] = by_store.shift(7)
    df["avg_last_7_days"] = by_store.transform(lambda s: s.shift(1).rolling(7).mean())
    return df.dropna().reset_index(drop=True)


def main() -> None:
    print("Step 1/5  Loading data")
    data = generate_sales_data()

    print("Step 2/5  Validating data")
    checks = validate(data)
    for name, passed in checks.items():
        print(f"          {'PASS' if passed else 'FAIL'}  {name}")
    if not all(checks.values()):
        sys.exit("Data validation failed. Stopping the pipeline.")

    print("Step 3/5  Building features and time-aware split")
    data = add_history_features(data)
    data_version = hashlib.md5(pd.util.hash_pandas_object(data).values.tobytes()).hexdigest()[:8]
    train = data[data["date"] < "2025-10-01"]
    test = data[data["date"] >= "2025-10-01"]

    print("Step 4/5  Training model")
    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=SEED)
    model.fit(make_features(train), train["sales"])

    print("Step 5/5  Acceptance gate (model must beat the baseline)")
    model_mae = mean_absolute_error(test["sales"], model.predict(make_features(test)))
    baseline_mae = mean_absolute_error(test["sales"], test["last_week_sales"])
    print(f"          Model MAE {model_mae:.2f}  vs  Baseline MAE {baseline_mae:.2f}")
    if model_mae >= baseline_mae:
        sys.exit("Model did not beat the baseline. Not releasing it.")

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.joblib")
    metadata = {
        "model_version": MODEL_VERSION,
        "data_version": data_version,
        "model_mae": round(model_mae, 2),
        "baseline_mae": round(baseline_mae, 2),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open("model/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model version {MODEL_VERSION} saved to model/  {metadata}")


if __name__ == "__main__":
    main()
