"""Optional FastAPI scoring service for Project FORESIGHT."""
from pathlib import Path
from fastapi import FastAPI, HTTPException
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "processed" / "risk_scoring_results.csv"
risk = pd.read_csv(DATA)
app = FastAPI(title="Project FORESIGHT Scoring Service", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok", "rows": int(len(risk))}

@app.get("/score/{sku}")
def score_sku(sku: str):
    rows = risk[risk["sku"].eq(sku)].copy()
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Unknown SKU: {sku}")
    rows["forecast_week"] = rows["forecast_week"].astype(str)
    return rows.to_dict(orient="records")
