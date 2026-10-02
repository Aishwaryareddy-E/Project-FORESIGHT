# 3–5 Minute Demo Script

## 0:00–0:30 — Problem
“Project FORESIGHT converts sales and inventory extracts into SKU-level demand forecasts and inventory decisions. The goal is to identify what should be reordered, cleared, watched or left alone.”

## 0:30–1:00 — Data foundation
Show `01_data_pipeline.ipynb` briefly. Point out the four raw extracts, data-quality checks, weekly aggregation, and time-aware inventory mapping.

## 1:00–1:45 — Forecasting
Show the forecasting notebook/report. Explain the Seasonal-Naive baseline, Holt comparison, feature engineering and four rolling-origin folds. State that HistGradientBoostingRegressor achieved 10.24% rolling WAPE versus 10.84% for the baseline.

## 1:45–2:30 — Risk scoring
Show `03_risk_scoring.ipynb`. Explain lead-time demand, progressive inventory, stockout/overstock thresholds, and the four-quadrant decisioning.

## 2:30–3:30 — Dashboard
Open the Streamlit app. Demonstrate:
- category filter
- SKU filter
- forecast vs actual
- risk charts
- decision grid
- reorder table
- markdown table
- SKU detail
- CSV export

## 3:30–4:15 — Business impact
Show the exposure KPIs and explain that they are planning estimates, not realized losses. Explain how the priority view helps operations focus on the highest-value decisions.

## 4:15–5:00 — Limitations & close
Mention the inventory timing assumption for on-order stock, the project-defined thresholds, and the absence of future inbound arrival dates. Finish by showing the repository structure and deployment target.
