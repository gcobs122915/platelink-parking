@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
"%~dp0..\HAILO_PLATE\.venv\Scripts\python.exe" -B -X utf8 -u "%~dp0app\client.py" --core-root "%~dp0." %*
set "plate_exit=%ERRORLEVEL%"
echo.
pause
exit /b %plate_exit%
