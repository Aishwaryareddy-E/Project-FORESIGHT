# Forecast Model Selection

The project uses four 13-week rolling-origin folds specified in the FORESIGHT workflow. WAPE is the primary metric.

| Method | Final holdout WAPE | Rolling average WAPE |
|---|---:|---:|
| Seasonal Naive | 11.62% | 10.84% |
| Holt Exponential Smoothing | 10.56% | 15.19% |
| HistGradientBoostingRegressor | 8.96% | **10.24%** |

The gradient-boosted model is selected for the production forecast because it has the lowest rolling-origin average WAPE. It uses lag, rolling, calendar/seasonality and promotion features, with recursive multi-week prediction to avoid using future actual demand during evaluation.

The result is an approximate **5.6% relative WAPE improvement** over the Seasonal-Naive baseline on the rolling-origin average.

The model is not described as a guarantee of future business performance; it is the best-performing candidate within this backtest design.
