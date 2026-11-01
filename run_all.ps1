# -------------------------------------------------
# GraphSentry AML GNN – One-click execution (PowerShell)
# -------------------------------------------------

$ErrorActionPreference = "Stop"

# 1. Activate or create virtual environment
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    $pyBin = if (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe") {
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
    } else {
        "python"
    }
    Write-Host "[*] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    & $pyBin -m venv .venv
}

$venvPython = "$PSScriptRoot\.venv\Scripts\python.exe"

# 2. Generate data & build graph if not present
if (-not (Test-Path "data\processed\hetero_graph.pt")) {
    Write-Host "[*] Generating synthetic AML data..." -ForegroundColor Yellow
    & $venvPython data/raw/generate_mock_aml_data.py --num-accounts 5000 --num-transactions 30000
    Write-Host "[*] Processing graph..." -ForegroundColor Yellow
    & $venvPython data/process_graph.py
}

# 3. Train model if not present
if (-not (Test-Path "artifacts\models\best_hetero_gnn.pt")) {
    Write-Host "[*] Training HeteroGraphSAGE model..." -ForegroundColor Yellow
    & $venvPython src/training/train.py --epochs 10 --batch-size 512
}

# 4. Start API server
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "GraphSentry AML Server Starting on http://localhost:8000" -ForegroundColor Green
Write-Host "Swagger Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Health Check: http://localhost:8000/health" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
& $venvPython -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
