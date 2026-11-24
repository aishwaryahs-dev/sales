"""Shared feature logic. Used by BOTH training (train.py) and serving (app.py),
so the model always sees features built the same way (no training-serving skew)."""
import pandas as pd

FEATURES = ["store_id", "day_of_week", "is_weekend", "month", "last_week_sales", "avg_last_7_days"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    out = pd.DataFrame({
        "store_id":        df["store_id"].astype(int),
        "day_of_week":     d.dt.dayofweek,
        "is_weekend":      (d.dt.dayofweek >= 5).astype(int),
        "month":           d.dt.month,
        "last_week_sales": df["last_week_sales"].astype(float),
        "avg_last_7_days": df["avg_last_7_days"].astype(float),
    })
    return out[FEATURES]
