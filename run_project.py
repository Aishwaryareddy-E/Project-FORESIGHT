"""Run the production FORESIGHT workflow.

The heavier four-fold ML experiment is kept as a separate reproducibility step:
`python src/ml_experiment.py`.
The production run rebuilds the data layer, final forecast, and risk outputs.
"""
from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parent
for script in ["src/pipeline.py", "src/final_forecast_ml.py", "src/risk.py"]:
    print(f"\n>>> Running {script}")
    subprocess.run([sys.executable, str(ROOT/script)], check=True)
print("\nFORESIGHT production pipeline completed.")
