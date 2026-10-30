"""FastAPI backend: loads ../best_model.pkl and forecasts cash vs online."""
import io
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from train import TARGETS as _TARGETS
from train import excel_to_daily, merge_append, train_bundle

app = FastAPI(title="Telecom Encaissement API")

BASE = Path(__file__).resolve().parent.parent
PKL = BASE / "best_model.pkl"
with open(PKL, "rb") as f:
    B = pickle.load(f)

MODELS = B["models"]
FEATURES = B["feature_cols"]
TARGETS = B["targets"]
DAILY = B["daily"]
BEST = B["best_names"]
METRICS = B.get("metrics", [])


def forecast_target(target: str, days: int):
    hist = DAILY.set_index("date")[target].copy()
    last = hist.index.max()
    dates, preds = [], []
    for i in range(1, days + 1):
        d = last + pd.Timedelta(days=i)
        lag_1 = hist.iloc[-1]
        lag_7 = hist.iloc[-7] if len(hist) >= 7 else hist.iloc[0]
        roll_7 = hist.iloc[-7:].mean() if len(hist) >= 7 else hist.mean()
        row = pd.DataFrame([{
            "trend": len(DAILY) - 1 + i, "dow": d.dayofweek,
            "month": d.month, "dom": d.day,
            "weekend": int(d.dayofweek >= 5),
            "lag_1": lag_1, "lag_7": lag_7, "roll_7": roll_7,
        }])[FEATURES]
        v = float(max(0, MODELS[target].predict(row)[0]))
        dates.append(d.strftime("%Y-%m-%d"))
        preds.append(v)
        hist.loc[pd.Timestamp(d)] = v
    return dates, preds


def _retrain_and_save(daily: pd.DataFrame) -> dict:
    """Train on daily, save bundle to disk, refresh in-memory globals."""
    global MODELS, FEATURES, TARGETS, DAILY, BEST, METRICS
    bundle = train_bundle(daily)
    with open(PKL, "wb") as f:
        pickle.dump(bundle, f)
    MODELS = bundle["models"]
    FEATURES = bundle["feature_cols"]
    TARGETS = bundle["targets"]
    DAILY = bundle["daily"]
    BEST = bundle["best_names"]
    METRICS = bundle["metrics"]
    d = DAILY.copy()
    return {"rows": len(d),
            "date_from": pd.to_datetime(d["date"].min()).strftime("%Y-%m-%d"),
            "date_to": pd.to_datetime(d["date"].max()).strftime("%Y-%m-%d"),
            "best_models": BEST, "metrics": METRICS}


@app.get("/")
def root():
    return {"ok": True, "best_models": BEST, "targets": TARGETS}


@app.get("/history")
def history():
    d = DAILY.copy()
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    return d.to_dict(orient="records")


@app.get("/predict")
def predict(days: int = Query(30, ge=1, le=90)):
    dates, _ = forecast_target(TARGETS[0], days)
    out = {"dates": dates}
    for t in TARGETS:
        _, p = forecast_target(t, days)
        out[t] = p
    return out


@app.post("/retrain/append")
async def retrain_append(file: UploadFile = File(...)):
    """Append newer Excel data: sum values on overlapping days, retrain on bigger dataset."""
    try:
        content = await file.read()
        new_daily = excel_to_daily(io.BytesIO(content))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Could not read Excel file: {e}")
    merged = merge_append(DAILY, new_daily)
    info = _retrain_and_save(merged)
    return {"mode": "append", **info}


@app.post("/retrain/replace")
async def retrain_replace(file: UploadFile = File(...)):
    """Replace old dataset with the uploaded Excel file, retrain only on new data."""
    try:
        content = await file.read()
        new_daily = excel_to_daily(io.BytesIO(content))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Could not read Excel file: {e}")
    info = _retrain_and_save(new_daily)
    return {"mode": "replace", **info}
