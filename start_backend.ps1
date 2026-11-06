#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Start the ATHLETEIQ FastAPI backend.
.DESCRIPTION
    Activates the virtual environment and launches uvicorn with hot-reload.
    Run from the repo root: .\start_backend.ps1
#>

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "  ATHLETEIQ – Backend Startup" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Activate venv
$activate = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $activate) {
    & $activate
    Write-Host "  ✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "  ⚠ No .venv found — using system Python" -ForegroundColor Yellow
}

Write-Host "  Starting FastAPI on http://localhost:8000" -ForegroundColor Green
Write-Host "  Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""

& uvicorn app.athleteiq_api:app --reload --host 0.0.0.0 --port 8000
