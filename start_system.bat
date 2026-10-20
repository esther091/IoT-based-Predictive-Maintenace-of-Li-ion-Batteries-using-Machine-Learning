@echo off
REM ============================================================
REM  start_system.bat
REM  Double-click to start the backend and the dashboard.
REM  Each opens in its own window; close a window to stop it.
REM
REM  It calls the project's own venv\Scripts\python.exe directly, so it
REM  works even though the project folder was moved after the venv was
REM  created (activate.bat and streamlit.exe still point to the old path).
REM ============================================================
cd /d "%~dp0"
set "PY=%~dp0venv\Scripts\python.exe"

if not exist "%PY%" (
    echo.
    echo  The virtual environment was not found. Set it up once with:
    echo      python -m venv venv
    echo      venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Starting the backend (FastAPI, port 8000) ...
start "Backend - FastAPI" /D "%~dp0backend" cmd /k ""%PY%" main.py"

REM give the backend time to load the models before the dashboard asks for data
timeout /t 8 /nobreak >nul

echo Starting the dashboard (Streamlit) ...
start "Dashboard - Streamlit" /D "%~dp0dashboard" cmd /k ""%PY%" -m streamlit run dashboard.py --theme.base light"

echo.
echo  Both are starting. The dashboard opens in your browser.
echo.
echo  Your laptop's IP address (the ESP32 firmware SERVER_URL must use this):
ipconfig | findstr /i "IPv4"
echo.
echo  SERVER_URL should look like:  http://THAT-IP:8000/readings
echo.
pause
