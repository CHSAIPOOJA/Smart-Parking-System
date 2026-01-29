# Smart Parking System - PowerShell Runner
# This script activates the virtual environment and runs main.py

$venvPath = ".\.venv\Scripts\python.exe"
$scriptPath = "code/main.py"

Write-Host "Starting Smart Parking System..." -ForegroundColor Green
Write-Host "Using Python: $venvPath" -ForegroundColor Cyan

& $venvPath $scriptPath

Write-Host "Smart Parking System stopped." -ForegroundColor Yellow
