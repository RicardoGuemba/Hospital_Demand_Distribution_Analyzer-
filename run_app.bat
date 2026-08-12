@echo off
cd /d "%~dp0"

where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Erro: Python 3.11+ nao encontrado.
    echo Instale em https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Criando ambiente virtual...
    python -m venv .venv
    call .venv\Scripts\pip install -r requirements.txt
    call .venv\Scripts\pip install -e .
)

echo Iniciando Hospital Demand Analyzer...
.venv\Scripts\python -m hospital_demand_analyzer
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% neq 0 (
    echo.
    echo Erro ao iniciar (codigo %EXIT_CODE%). Veja logs\app.log
    pause
)

exit /b %EXIT_CODE%
