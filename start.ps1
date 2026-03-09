# PowerShell start script for Windows

Write-Host "🚀 Starting CKD Prediction System..." -ForegroundColor Cyan

# Check if in correct directory
if (-not (Test-Path "docker-compose.yml")) {
    Write-Host "❌ Please run this script from the project root directory" -ForegroundColor Red
    exit 1
}

# Start backend
Write-Host "📦 Starting Backend Server..." -ForegroundColor Yellow
Set-Location server

if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

.\venv\Scripts\Activate.ps1
pip install -r requirements.txt *> $null

# Create logs directory
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

# Start backend
Write-Host "Starting Flask API on http://localhost:5000" -ForegroundColor Green
$backend = Start-Process python -ArgumentList "app.py" -PassThru -WindowStyle Hidden
Set-Location ..

Start-Sleep -Seconds 3

# Start frontend
Write-Host "📦 Starting Frontend Server..." -ForegroundColor Yellow
Set-Location client

if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    npm install
}

Write-Host "Starting React app on http://localhost:5173" -ForegroundColor Green
$frontend = Start-Process npm -ArgumentList "run dev" -PassThru -WindowStyle Normal
Set-Location ..

Write-Host ""
Write-Host "✅ System is running!" -ForegroundColor Green
Write-Host "   Frontend: http://localhost:5173" -ForegroundColor White
Write-Host "   Backend:  http://localhost:5000" -ForegroundColor White
Write-Host "   API Docs: http://localhost:5000" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop (may need to kill processes manually)" -ForegroundColor Yellow
Write-Host "Backend PID: $($backend.Id)" -ForegroundColor Gray
Write-Host "Frontend PID: $($frontend.Id)" -ForegroundColor Gray
Write-Host ""

# Wait for user input
Read-Host "Press Enter to stop services"

# Cleanup
Write-Host "🛑 Stopping services..." -ForegroundColor Yellow
Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue
Write-Host "✅ Stopped" -ForegroundColor Green
