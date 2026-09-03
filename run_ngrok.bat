@echo off
setlocal
cd /d "%~dp0"
if "%NGROK_AUTHTOKEN%"=="" (
  echo NGROK_AUTHTOKEN is not set.
  echo Set it in your environment, then run this file again.
  echo Example: setx NGROK_AUTHTOKEN "YOUR_TOKEN"
  pause
  exit /b 1
)
python run_local.py ngrok
pause
