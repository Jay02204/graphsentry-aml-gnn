@echo off
setlocal enabledelayedexpansion
rem -------------------------------------------------
rem GraphSentry AML GNN – One‑click execution (Windows)
rem -------------------------------------------------

:: 1️⃣ Activate or create virtual environment
if not exist .venv\Scripts\activate.bat (
    if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
        set "PY_BIN=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    ) else (
        set "PY_BIN=python"
    )
    echo [*] Creating virtual environment using !PY_BIN!...
    "!PY_BIN!" -m venv .venv
)
call .venv\Scripts\activate.bat

:: 2️⃣ Generate mock data & build graph if not present
if not exist data\processed\hetero_graph.pt (
    echo [*] Generating mock AML data...
    python data\raw\generate_mock_aml_data.py --num-accounts 5000 --num-transactions 30000
    echo [*] Processing graph...
    python data\process_graph.py
)

:: 3️⃣ Train model if not present
if not exist artifacts\models\best_hetero_gnn.pt (
    echo [*] Training model...
    python src\training\train.py --epochs 10 --batch-size 512
)

:: 4️⃣ Start FastAPI server
echo =======================================================
echo GraphSentry AML Server Starting on http://localhost:8000
echo Swagger Docs: http://localhost:8000/docs
echo Health Check: http://localhost:8000/health
echo =======================================================
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
