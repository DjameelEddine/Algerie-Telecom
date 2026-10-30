"""Shared training logic: Excel -> daily table -> train -> bundle. Used by backend retrain endpoints."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGETS = ["cash_amount", "online_amount", "cash_count", "online_count"]
FEATURES = ["trend", "dow", "month", "dom", "weekend", "lag_1", "lag_7", "roll_7"]


def excel_to_daily(source) -> pd.DataFrame:
    """Parse an Excel file (same 12-column monthly-sheet layout) into a daily table."""
    xls = pd.ExcelFile(source)
    frames = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        df = df.iloc[:, :12]
        df.columns = ["method", "cash_raw", "timbre", "total", "cheque", "virement",
                      "autres", "no_cheque", "code_banque", "dir_banque", "date", "dept"]
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)

    df = raw.copy()
    m = df["method"].astype(str).str.lower().str.strip()
    is_online = m.str.contains("ligne|paiment|paiemnt", na=False)
    is_cash = m.str.contains("esp", na=False)
    df["pay"] = "other"
    df.loc[is_online, "pay"] = "online"
    df.loc[is_cash, "pay"] = "cash"
    df = df[df["pay"].isin(["cash", "online"])]
    if df.empty:
        raise ValueError("No cash/online rows found in uploaded file.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True).dt.normalize()
    df = df.dropna(subset=["date"])
    if df.empty:
        raise ValueError("No parseable dates found in uploaded file.")

    for c in ["total", "autres"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["cash_amount"] = 0.0
    df["online_amount"] = 0.0
    df.loc[df["pay"] == "cash", "cash_amount"] = df.loc[df["pay"] == "cash", "total"]
    df.loc[df["pay"] == "online", "online_amount"] = df.loc[df["pay"] == "online", "autres"]

    pv_cash = df[df["pay"] == "cash"].pivot_table(index="date", values="cash_amount", aggfunc="sum")
    pv_online = df[df["pay"] == "online"].pivot_table(index="date", values="online_amount", aggfunc="sum")
    pv_count = df.pivot_table(index="date", columns="pay", aggfunc="size", fill_value=0)

    daily = pd.DataFrame(index=pd.date_range(df["date"].min(), df["date"].max(), freq="D"))
    daily.index.name = "date"
    daily["cash_amount"] = pv_cash.reindex(daily.index, fill_value=0)
    daily["online_amount"] = pv_online.reindex(daily.index, fill_value=0)
    daily["cash_count"] = pv_count["cash"].reindex(daily.index, fill_value=0) if "cash" in pv_count else 0
    daily["online_count"] = pv_count["online"].reindex(daily.index, fill_value=0) if "online" in pv_count else 0
    return daily.reset_index()


def merge_append(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append mode: outer-join on date, SUM values on overlapping days, fill gaps with 0."""
    both = pd.concat([old, new], ignore_index=True)
    both["date"] = pd.to_datetime(both["date"]).dt.normalize()
    merged = both.groupby("date", as_index=False)[TARGETS].sum()
    full = pd.DataFrame({"date": pd.date_range(merged["date"].min(), merged["date"].max(), freq="D")})
    merged = full.merge(merged, on="date", how="left").fillna(0)
    return merged.sort_values("date").reset_index(drop=True)


def add_features(daily_df: pd.DataFrame, target: str) -> pd.DataFrame:
    d = daily_df[["date", target]].copy()
    d["trend"] = np.arange(len(d))
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    d["dom"] = d["date"].dt.day
    d["weekend"] = (d["dow"] >= 5).astype(int)
    d["lag_1"] = d[target].shift(1)
    d["lag_7"] = d[target].shift(7)
    d["roll_7"] = d[target].shift(1).rolling(7).mean()
    return d.dropna().reset_index(drop=True)


def train_bundle(daily: pd.DataFrame) -> dict:
    """Train 4 models x 4 targets (80/20 chronological split, winner = lowest MAE), refit winners on all data."""
    daily = daily.sort_values("date").reset_index(drop=True)
    results = []
    best_names = {}
    best_models = {}
    for target in TARGETS:
        feat = add_features(daily, target)
        n = len(feat)
        split = int(n * 0.8)
        X, y = feat[FEATURES], feat[target]
        Xtr, Xte = X.iloc[:split], X.iloc[split:]
        ytr, yte = y.iloc[:split], y.iloc[split:]
        cands = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
        }
        scored = []
        for name, model in cands.items():
            model.fit(Xtr, ytr)
            p = model.predict(Xte)
            r = {"target": target, "model": name,
                 "MAE": float(mean_absolute_error(yte, p)),
                 "RMSE": float(mean_squared_error(yte, p) ** 0.5),
                 "R2": float(r2_score(yte, p))}
            results.append(r)
            scored.append(r)
        winner = sorted(scored, key=lambda r: r["MAE"])[0]["model"]
        best_names[target] = winner
        final = {"LinearRegression": LinearRegression(), "Ridge": Ridge(),
                 "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
                 "GradientBoosting": GradientBoostingRegressor(random_state=42)}[winner]
        final.fit(X, y)
        best_models[target] = final
    return {"models": best_models, "best_names": best_names, "metrics": results,
            "feature_cols": FEATURES, "daily": daily, "targets": TARGETS}
