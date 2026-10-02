# Deployment Guide

## Local dashboard

From the repository root:

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The dashboard uses only files under `data/processed/` at runtime.

## Streamlit Community Cloud

1. Create a GitHub repository for this project.
2. Do **not** publish `data/raw/`; it is excluded by `.gitignore`.
3. Commit `app.py`, `data/processed/`, `src/`, `reports/`, `requirements.txt`, and notebooks.
4. Create a Streamlit Community Cloud app pointing to `app.py`.
5. Smoke-test the public URL with the seeded processed data.

## Optional FastAPI service

```powershell
uvicorn service:app --host 0.0.0.0 --port 8000
```

Test:

```text
GET /health
GET /score/SKU001
```

For a public hosted endpoint, deploy `service.py` to a compatible Python web host. The project package contains the endpoint contract; the actual public hosting step requires the owner's hosting account and credentials.
