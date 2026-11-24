"""Model serving: a REST API with a clear input and output contract,
a health check, and a simple monitoring endpoint."""
import datetime
import json
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from features import make_features

model = joblib.load("model/model.joblib")
with open("model/metadata.json") as f:
    metadata = json.load(f)

app = FastAPI(title="Store Sales Forecast API", version=metadata["model_version"])

stats = {"predictions_served": 0, "total_latency_ms": 0.0, "total_prediction": 0.0}


class SalesRequest(BaseModel):          # input schema + validation
    store_id: int = Field(ge=1, le=5, examples=[3])
    date: datetime.date = Field(examples=["2026-01-03"])
    last_week_sales: float = Field(ge=0, examples=[230])
    avg_last_7_days: float = Field(ge=0, examples=[185])


class SalesResponse(BaseModel):         # output schema
    prediction: float
    model_version: str
    request_id: str


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    """Used by Render (and CI) to check the service is up."""
    return {"status": "ok", "model_version": metadata["model_version"]}


@app.post("/predict", response_model=SalesResponse)
def predict(req: SalesRequest):
    start = time.perf_counter()
    features = make_features(pd.DataFrame([req.model_dump()]))
    prediction = round(float(model.predict(features)[0]), 1)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    result = {"prediction": prediction,
              "model_version": metadata["model_version"],
              "request_id": str(uuid.uuid4())[:8]}

    stats["predictions_served"] += 1
    stats["total_latency_ms"] += latency_ms
    stats["total_prediction"] += prediction
    # One log line per prediction. These show up in the Render Logs page.
    print(json.dumps({"event": "prediction", **result, "store_id": req.store_id,
                      "date": str(req.date), "latency_ms": latency_ms}), flush=True)
    return result


@app.get("/monitoring")
def monitoring():
    """Simple live monitoring: how much traffic, how fast, which model."""
    n = stats["predictions_served"]
    return {
        "model_version": metadata["model_version"],
        "data_version": metadata["data_version"],
        "test_mae": metadata["model_mae"],
        "baseline_mae": metadata["baseline_mae"],
        "predictions_served": n,
        "avg_latency_ms": round(stats["total_latency_ms"] / n, 2) if n else None,
        "avg_prediction": round(stats["total_prediction"] / n, 1) if n else None,
    }
