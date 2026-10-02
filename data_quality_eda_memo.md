# Project FORESIGHT — Data Quality & EDA Memo

## Executive summary

The project uses four simulated D2C extracts covering daily sales, SKU master data, calendar attributes, and periodic inventory snapshots. The cleaned pipeline produces **5,250 weekly SKU records across 50 SKUs**, with complete SKU-week coverage and no future inventory mappings.

## Data quality findings

| Finding | Result | Treatment |
|---|---:|---|
| Sales rows | 36,550 | Retained |
| Sales SKUs | 50 | Retained |
| Sales dates | 731 | Complete daily coverage |
| Duplicate SKU-date rows | 0 | No removal required |
| Revenue formula mismatches | 0 | Validated |
| Negative gross-margin rows | 11,696 | Retained; treated as valid business information |
| Sales before recorded launch date | 3,393 | Retained and flagged as metadata inconsistency |
| Calendar blank holiday labels | 723 | Replaced with `No Holiday` |
| Calendar blank promotion labels | 656 | Replaced with `No Promotion` |
| Inventory SKUs | 200 | Filtered to 50 project SKUs |
| Inventory snapshots per project SKU | 24 | Complete |
| Inventory value formula mismatches | 1,176 | Reported; valuation formula not changed because the source definition is unspecified |

## Business insights

1. **Home Decor is the largest category by units**, with 132,570 units sold in the analysis period.
2. **SKU012 is the top unit seller** with 19,067 units, while **SKU026 is the top revenue SKU** with approximately ₹180.9M revenue.
3. A strict dead-stock rule (positive latest on-hand inventory and zero demand over the previous 13 complete weeks) returned **0 candidates**; therefore, the analysis uses slow-mover coverage to surface inventory exposure.
4. **SKU025** is the most extreme slow mover under the latest-snapshot / recent-demand coverage definition, with 1,124 units on hand and about 74 weeks of estimated coverage.
5. The demand series shows clear weekly and seasonal variation, so a seasonal baseline is appropriate and was explicitly benchmarked.

## Forecasting findings

The forecasting workflow uses a seasonal-naive baseline, Holt exponential smoothing, and a feature-based gradient-boosted tree experiment. The final selection rule is the four-fold rolling-origin average WAPE.

| Method | Single holdout WAPE | Rolling-origin average WAPE | Role |
|---|---:|---:|---|
| Seasonal Naive | 11.62% | 10.84% | Baseline / fallback |
| Holt Exponential Smoothing | 10.56% | 15.19% | Classical comparison |
| HistGradientBoostingRegressor | 8.96% | **10.24%** | Selected final model |

The feature-based model improves rolling-origin WAPE by about **5.6% relative to the Seasonal-Naive baseline**. The model is therefore used for the final 13-week forecast.

## Risk & decisioning findings

Risk scoring uses lead-time demand and a progressive inventory position. The model assumes the latest known on-hand plus on-order quantity forms the starting inventory position because exact inbound arrival dates are not available.

- High stockout risk: 390 of 650 forecast records
- Medium stockout risk: 40
- High overstock risk: 50
- Medium overstock risk: 30
- Reorder: 390
- Markdown/Clear: 50
- Monitor/Healthy: 210
- Watch/Volatile: 0 under the current rules

Estimated exposure across the forecast horizon:

- Potential lost-sales value: approximately **₹312.9M**
- Potential overstock inventory value: approximately **₹38.1M**
- Estimated reorder value: approximately **₹981.5M**

These are model-based exposure estimates, not realized financial losses.

## Limitations

- Future inbound arrival dates are not supplied; on-order stock is therefore carried as a starting inventory assumption.
- Risk thresholds are transparent project business rules, not learned probabilities.
- Inventory valuation inconsistencies are reported rather than overwritten.
- The final forecast is a point forecast; prediction intervals were not implemented in the core deliverable.
