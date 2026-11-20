@echo off
setlocal EnableExtensions
title TARO Ubuntu Jupyter Server

echo ================================================
echo Starting the Ubuntu WSL Jupyter server
echo ================================================
echo.

where wsl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: wsl.exe was not found. Install WSL and Ubuntu first.
    pause
    exit /b 1
)

set "PROJECT_WIN=%~dp0"
if "%PROJECT_WIN:~-1%"=="\" set "PROJECT_WIN=%PROJECT_WIN:~0,-1%"
for /f "delims=" %%P in ('wsl.exe -d Ubuntu -- wslpath -a -u "%PROJECT_WIN%" 2^>nul') do set "PROJECT_WSL=%%P"
if not defined PROJECT_WSL (
    echo ERROR: The Ubuntu distribution was not found or the project path could not be converted.
    echo Check the installed distribution with: wsl --list --verbose
    pause
    exit /b 1
)

echo Project folder: %PROJECT_WIN%
echo Ubuntu folder: %PROJECT_WSL%
echo.
echo Keep this window open while you use the notebook server.
echo Open the URL printed below in your Windows browser.
echo.

wsl.exe -d Ubuntu -- bash -lc "cd -- '%PROJECT_WSL%' || exit 1; if [ -x \"\$HOME/taro-venv/bin/jupyter\" ]; then JUPYTER=\"\$HOME/taro-venv/bin/jupyter\"; elif command -v jupyter >/dev/null 2>&1; then JUPYTER=\"\$(command -v jupyter)\"; else echo 'ERROR: Jupyter is not installed in Ubuntu.'; echo 'Run the standalone notebook setup first, or install it with: python3 -m pip install jupyter'; exit 1; fi; exec \"\$JUPYTER\" notebook --no-browser --ip=127.0.0.1 --notebook-dir='%PROJECT_WSL%'"

echo.
echo The Ubuntu server has stopped.
pause
