https://project-foresight-u7kzcbejc8kyv4a86eapp7k.streamlit.app/

# Project FORESIGHT — Demand & Inventory Intelligence

A reproducible demand forecasting, inventory risk scoring, decisioning and dashboard project built for the Zidio FORESIGHT client engagement.

## What this project delivers

- Reproducible raw-to-weekly data pipeline
- Data quality checks and EDA
- SKU-level weekly demand forecasting
- Seasonal-Naive baseline + Holt comparison
- Feature-based HistGradientBoosting experiment
- Rolling-origin backtesting using WAPE
- Lead-time stockout risk scoring
- Overstock coverage scoring
- Four-quadrant decisioning
- Reorder / markdown recommendations
- Rupee exposure estimates
- Streamlit planning dashboard
- Optional FastAPI scoring endpoint
- Executive readout materials

## Final forecasting result

| Method | Rolling WAPE |
|---|---:|
| Seasonal Naive | 10.84% |
| Holt Exponential Smoothing | 15.19% |
| HistGradientBoostingRegressor | **10.24%** |

The gradient-boosted model is the operational forecast because it performed best on the specified four-fold rolling-origin backtest.

## Project structure

```text
Project_FORESIGHT/
├── app.py
├── service.py
├── run_project.py
├── run_dashboard.bat
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                    # local/private raw extracts
│   └── processed/              # generated project outputs
├── src/
│   ├── pipeline.py
│   ├── ml_experiment.py
│   ├── final_forecast_ml.py
│   └── risk.py
├── notebooks/
│   ├── 01_data_pipeline.ipynb
│   ├── 02_forecasting.ipynb
│   └── 03_risk_scoring.ipynb
└── reports/
    ├── data_quality_eda_memo.md
    ├── model_comparison.csv
    ├── rolling_backtest_results.csv
    ├── risk_summary.json
    ├── forecast_selection_summary.json
    ├── figures/
    └── executive_readout.pptx
```

## Run the core pipeline

From the project root:

```powershell
pip install -r requirements.txt
python run_project.py
```

This regenerates:

- `data/processed/analysis_ready_weekly.csv`
- `data/processed/demand_forecast_results.csv`
- `data/processed/forecast_evaluation_results.csv`
- `data/processed/risk_scoring_results.csv`

## Run the dashboard

```powershell
streamlit run app.py
```

Or double-click `run_dashboard.bat` on Windows.

## Optional scoring API

```powershell
uvicorn service:app --reload
```

Endpoints:

- `GET /health`
- `GET /score/{sku}`

## Dashboard features

- Category and SKU filters
- Stockout risk filtering
- Decision/quadrant filtering
- Forecast vs actual backtest view
- Stockout / overstock distributions
- SKU-level four-quadrant decision grid
- Prioritized reorder list
- Prioritized markdown/clear list
- SKU scoring detail
- CSV export of filtered results

## Important assumptions

1. Inventory position starts from latest known on-hand + on-order stock.
2. Exact future inbound arrival dates are not available, so on-order stock is treated as available within the planning horizon.
3. Stockout risk uses estimated lead-time demand from the weekly forecast and the recorded lead time.
4. Overstock thresholds (>13 weeks High, >8 weeks Medium) are project-defined business rules.
5. Rupee figures are exposure estimates, not realized financial results.

## Data governance

The raw files are treated as local/private inputs. Do not publish the raw client extracts to a public Git repository. The supplied engagement brief explicitly treats the simulated client data as confidential.

## Project honesty note

The model-selection process is intentionally transparent: the Seasonal-Naive baseline, Holt model and feature-based ML candidate are all documented, and the final model is chosen from rolling-origin performance rather than from a preferred method.
