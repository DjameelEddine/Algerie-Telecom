# Technical details — Telecom Encaissement forecast

## Data
- File: `encaissement du 01-01-2026.xlsx`, 8 sheets (`janvier`..`aout`), ~627k rows total.
- Columns vary slightly per sheet, so code uses column positions, renamed to: `method, cash_raw, timbre, total, cheque, virement, autres, no_cheque, code_banque, dir_banque, date, dept`.
- Key fact: cash rows (`Especes`) store money in `Montant Total`; online rows (`paiement en ligne` + typos `paiement e ligne`, `paiment en ligne`, `paiemnt en ligne`) store money in `Autres` with `Montant Total = 0`.
- Method normalized to `cash` / `online` by substring match (`esp` vs `ligne|paiment|paiemnt`). Other/NaN rows dropped.
- Dates parsed with `pd.to_datetime(dayfirst=True)`, normalized to day. August has time strings, handled by the same parser.
- Result: 2026-01-01 → 2026-08-31, 243 calendar days (missing days filled with 0).

## Daily time series (what is forecast)
Per day, from `notebook.ipynb`:
- `cash_amount` = sum of `total` on cash rows
- `online_amount` = sum of `autres` on online rows
- `cash_count`, `online_count` = row counts per method
- Totals: cash ~551M DZD (617,757 txns), online ~16M DZD (9,948 txns). Online grows June→August.

## Features and models
- Features per target (8): `trend, dow, month, dom, weekend, lag_1, lag_7, roll_7`.
- 4 targets × 4 models: `LinearRegression, Ridge, RandomForest(200 trees), GradientBoosting`.
- Split: chronological 80/20. Metrics: MAE, RMSE, R². Winner per target = lowest MAE.
- Winners on this data: `cash_amount: GradientBoosting`, `online_amount: LinearRegression`, `cash_count: GradientBoosting`, `online_count: LinearRegression`.
- Winners refit on full history and saved.

## Forecast
- Recursive: to predict day `t+i`, build features from calendar + last predictions (`lag_1`, `lag_7`, `roll_7`), predict, clip at 0, append to history for the next step.
- Default horizon: 30 days (API allows 1–90).

## `best_model.pkl` bundle
`{models, best_names, metrics, feature_cols, daily, targets}` — everything the backend needs, no retraining at serve time.

## App (`app/`)
- `app/train.py` (shared logic): `excel_to_daily()` parses an Excel file (same 12-column monthly-sheet layout) into a daily table; `merge_append()` outer-joins old + new on date and SUMS values on overlapping days; `train_bundle()` trains 4 models × 4 targets and returns the same bundle as the notebook.
- `app/backend.py` (FastAPI): loads `../best_model.pkl` once. Endpoints: `GET /` (health + winners), `GET /history` (daily rows), `GET /predict?days=30` (forecasts), plus retraining: `POST /retrain/append` (upload newer Excel → sum on overlap → retrain on bigger dataset → overwrite `best_model.pkl`), `POST /retrain/replace` (upload newer Excel → discard old data → retrain only on new file → overwrite `best_model.pkl`). Both return `{mode, rows, date_from, date_to, best_models, metrics}`.
- `app/app.py` (Streamlit web app): hero header + intro text, sidebar settings (horizon slider), KPI cards (history days, cash collected, cash/online forecasts), forecast charts (cash history+forecast, online history+forecast, forecast comparisons), forecast table with CSV download, "Update the model" section with two upload cards (Append + Retrain, Replace + Retrain) that POST the file to the backend, and an "About the model" section.
- `app/requirements.txt`: `fastapi, uvicorn, streamlit, pandas, openpyxl, scikit-learn, matplotlib, requests`. Uploads need `python-multipart` on the backend (`pip install python-multipart`).

## Run (in `telecom` venv)
```
./telecom/bin/uvicorn backend:app --reload --app-dir app
./telecom/bin/streamlit run app/app.py
```
Backend runs on `http://127.0.0.1:8000`; frontend expects it there (`BACKEND` constant in `app/app.py`).
