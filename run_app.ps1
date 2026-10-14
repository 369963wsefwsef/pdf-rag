# 一键启动网页界面：自动使用项目内的 .venv
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "没找到 .venv，请先执行：python -m venv .venv" -ForegroundColor Red
    exit 1
}
Write-Host "启动中…… 浏览器会自动打开 http://localhost:8501" -ForegroundColor Green
& $python -m streamlit run app.py
