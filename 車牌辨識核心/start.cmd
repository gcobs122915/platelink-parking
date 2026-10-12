@echo off
setlocal
chcp 65001 >nul
title USB Plate Recognition - File Exchange
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
"%~dp0..\HAILO_PLATE\.venv\Scripts\python.exe" -B -X utf8 -u "%~dp0app\run.py" --core-root "%~dp0." %*
set "plate_exit=%ERRORLEVEL%"
echo.
echo File core stopped. Exit code: %plate_exit%
pause
exit /b %plate_exit%
