# Project FORESIGHT — Final QA Checklist

## Data foundation

- [x] Four raw extracts present locally
- [x] Reproducible pipeline reads all four raw files
- [x] 5,250 analysis-ready weekly rows
- [x] 50 SKUs
- [x] 5,250 unique SKU-week pairs
- [x] No future inventory mappings
- [x] Data-quality findings documented

## Forecasting

- [x] Seasonal-Naive baseline
- [x] Holt comparison
- [x] Feature-based ML model
- [x] Four rolling-origin folds
- [x] WAPE comparison
- [x] Final model selected from rolling performance
- [x] 13-week future forecast

## Risk & decisioning

- [x] Lead-time demand calculated
- [x] Progressive inventory position calculated
- [x] Stockout risk
- [x] Overstock risk
- [x] Four-quadrant decisioning
- [x] Reorder quantity/value
- [x] Markdown exposure
- [x] Priority levels

## Productization

- [x] Streamlit app source
- [x] SKU/category/risk/decision filters
- [x] Forecast vs actual view
- [x] Decision grid
- [x] Priority action tables
- [x] CSV export
- [x] Optional scoring API
- [ ] Public deployment smoke-test (requires owner account)

## Submission

- [x] README
- [x] Data-quality/EDA memo
- [x] Executive readout deck
- [x] Demo outline can be derived from the dashboard
- [ ] Final public URLs
