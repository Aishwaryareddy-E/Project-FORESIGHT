"""Project FORESIGHT forecast entry point.

Runs the feature-based ML backtest and then builds the final 13-week forecast.
"""
from pathlib import Path
import subprocess, sys
ROOT = Path(__file__).resolve().parents[1]
for script in ["ml_experiment.py", "final_forecast_ml.py"]:
    subprocess.run([sys.executable, str(ROOT / "src" / script)], check=True)
print("Forecasting stage completed.")
