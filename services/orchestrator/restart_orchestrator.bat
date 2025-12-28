@echo off
REM Script to restart Orchestrator service with new code

echo ================================================================================
echo Stopping old Orchestrator processes...
echo ================================================================================

REM Kill any existing uvicorn/orchestrator processes
taskkill /F /IM uvicorn.exe 2>nul
taskkill /F /FI "WINDOWTITLE eq *orchestrator*" 2>nul

timeout /t 2 /nobreak >nul

echo.
echo ================================================================================
echo Starting Orchestrator service...
echo ================================================================================

cd /d "c:\Users\admin\Downloads\Khiem\Chatbot-UIT\services\orchestrator"

REM Activate conda and start server
call conda activate chatbot-uit
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001

pause
