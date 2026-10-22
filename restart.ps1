# Stop any lingering python or streamlit processes to clear locks/ports
Write-Host "Stopping old bot and UI processes..." -ForegroundColor Yellow
Stop-Process -Name "python", "streamlit" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Start the LiveKit Bot Worker in a new terminal window
Write-Host "Starting Bot Worker..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", '& "C:\Users\harsi\AppData\Local\Programs\Python\Python314\python.exe" bot.py dev'

# Start the Streamlit UI in a new terminal window
Write-Host "Starting Streamlit UI..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", 'streamlit run app.py'

Write-Host "Everything is up and running! Open http://localhost:8501 in your browser." -ForegroundColor Magenta