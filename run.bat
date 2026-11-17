@echo off
setlocal
cd /d "%~dp0"
title Pressure Performance app

rem --- find Python (the "py" launcher first, then "python") ---
set PY=
py -3 --version >nul 2>&1 && set PY=py -3
if "%PY%"=="" (python --version >nul 2>&1 && set PY=python)
if "%PY%"=="" (
  echo Python was not found. Install Python 3.10 to 3.13 from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" in the installer, then run this file again.
  pause & exit /b 1
)
echo Using: %PY%
%PY% --version

rem --- private environment so nothing else on this PC is touched ---
if not exist .venv\Scripts\python.exe (
  echo Creating environment...
  %PY% -m venv .venv || (echo Could not create the environment. & pause & exit /b 1)
)
echo Installing packages (first run takes a few minutes)...
.venv\Scripts\python.exe -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo.
  echo Package install failed. Please copy the message above and send it back.
  pause & exit /b 1
)

echo.
echo Starting the app. Your browser will open at http://localhost:8501
echo Keep this window open while you use the app. Close it to stop.
start "" http://localhost:8501
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
echo.
echo The app stopped. See any error above.
pause
