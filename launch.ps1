Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Automated Setup & Launch Script " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# 1. Force stop old hanging python and streamlit processes
Write-Host "`n[1/3] Terminating existing background processes..." -ForegroundColor Yellow
Stop-Process -Name "python", "streamlit" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# 2. Validate .env configuration file exists
Write-Host "[2/3] Validating .env configuration file..." -ForegroundColor Yellow
if (!(Test-Path ".env")) {
    Write-Host "Error: .env file not found! Please create one with your credentials." -ForegroundColor Red
    exit
}
Write-Host "-> Environment file found and verified!" -ForegroundColor Green

# 3. Launch the Agent Worker and Streamlit App in separate isolated windows
Write-Host "[3/3] Launching LiveKit Worker & Streamlit UI..." -ForegroundColor Yellow

$pythonExe = "C:\Users\harsi\AppData\Local\Programs\Python\Python314\python.exe"

# Launch Agent
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host '--- LiveKit Worker Logs ---' -ForegroundColor Cyan; & '$pythonExe' bot.py dev"

# Wait for worker initialization
Start-Sleep -Seconds 3

# Launch UI
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host '--- Streamlit UI Logs ---' -ForegroundColor Green; streamlit run app.py"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " System Launch Complete! " -ForegroundColor Green
Write-Host " Open http://localhost:8501 in your browser. " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan