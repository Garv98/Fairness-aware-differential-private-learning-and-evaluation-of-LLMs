@echo off
REM FairDP demo launcher (Windows). First run trains the demo models (~3 min).
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [setup] creating Python environment...
  python -m venv .venv && .venv\Scripts\python -m pip install -r requirements.txt fastapi "uvicorn[standard]"
)
if not exist "server\artifacts\evaluation.json" (
  echo [setup] training demo models...
  pushd server && ..\.venv\Scripts\python train_models.py && popd
)
if not exist "web\node_modules" (
  echo [setup] installing web dependencies...
  pushd web && call npm install && popd
)
if not exist "web\.next\BUILD_ID" (
  pushd web && call npm run build && popd
)
start "FairDP API" cmd /k "cd /d %~dp0server && ..\.venv\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 8000"
start "FairDP Web" cmd /k "cd /d %~dp0web && npm run start"
timeout /t 6 >nul
start "" http://127.0.0.1:3000
echo Demo running: dashboard http://127.0.0.1:3000  ^|  API http://127.0.0.1:8000/docs
