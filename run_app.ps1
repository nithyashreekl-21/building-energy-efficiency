# PowerShell script to launch CoolBuild AI Streamlit application
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " CoolBuild AI - Intelligent Building Energy Optimizer" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting Streamlit application..." -ForegroundColor Yellow
& "C:\Users\shobh\AppData\Local\Python\bin\python.exe" -m streamlit run "$PSScriptRoot\app.py"
