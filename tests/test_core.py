from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def test_processed_outputs():
    analysis=pd.read_csv(ROOT/'data/processed/analysis_ready_weekly.csv')
    forecast=pd.read_csv(ROOT/'data/processed/demand_forecast_results.csv')
    risk=pd.read_csv(ROOT/'data/processed/risk_scoring_results.csv')
    assert analysis.shape[0] == 5250
    assert analysis['sku'].nunique() == 50
    assert analysis[['sku','week_start']].drop_duplicates().shape[0] == 5250
    assert forecast.shape[0] == 650
    assert forecast['sku'].nunique() == 50
    assert risk.shape[0] == 650
    assert risk[['sku','forecast_week']].drop_duplicates().shape[0] == 650
    assert risk['recommended_action'].notna().all()
    assert risk['decision_quadrant'].notna().all()
    assert (risk['inventory_coverage_weeks'] < 0).sum() == 0
