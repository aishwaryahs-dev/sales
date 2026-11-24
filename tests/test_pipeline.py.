"""Pipeline tests: unit, data, model and integration. Run with: pytest -v"""
import json
import os
import sys

import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import make_features  # noqa: E402
from train import generate_sales_data, validate  # noqa: E402


def test_unit_saturday_is_weekend():
    saturday = pd.DataFrame({"date": ["2026-01-03"], "store_id": [1],
                             "last_week_sales": [100], "avg_last_7_days": [100]})
    assert make_features(saturday)["is_weekend"].iloc[0] == 1


def test_data_passes_quality_checks():
    assert all(validate(generate_sales_data()).values())


def test_model_beats_baseline():
    with open("model/metadata.json") as f:
        meta = json.load(f)
    assert meta["model_mae"] < meta["baseline_mae"]


def test_api_health_and_predict():
    from app import app
    api = TestClient(app)
    assert api.get("/health").json()["status"] == "ok"
    response = api.post("/predict", json={"store_id": 3, "date": "2026-01-03",
                                          "last_week_sales": 230, "avg_last_7_days": 185})
    assert response.status_code == 200
    assert set(response.json()) == {"prediction", "model_version", "request_id"}


def test_api_rejects_bad_input():
    from app import app
    api = TestClient(app)
    response = api.post("/predict", json={"store_id": 3, "date": "2026-01-03",
                                          "last_week_sales": -10, "avg_last_7_days": 185})
    assert response.status_code == 422
