@echo off
title Tiny Voice Command Model (VCM) - Master Launcher
cd /d "%~dp0"

:MENU
cls
echo ==============================================================================
echo              TINY VOICE COMMAND MODEL (VCM) - MASTER LAUNCHER
echo ==============================================================================
echo  [1] Open GUI Control Center (VCM_Control_Center.exe)
echo  [2] Start JupyterLab Notebook Server (Port 8890)
echo  [3] Start PC Voice Assistant Live Demo (Port 7861)
echo  [4] Start Human Voice Dataset Recorder (Port 7862)
echo  [5] Stop Running Services on Ports 7861, 7862, 8890
echo  [6] Open All Web Dashboards in Browser
echo  [7] Exit
echo ==============================================================================
set /p choice="Enter choice [1-7]: "

if "%choice%"=="1" goto GUI
if "%choice%"=="2" goto JUPYTER
if "%choice%"=="3" goto DEMO
if "%choice%"=="4" goto RECORDER
if "%choice%"=="5" goto STOP
if "%choice%"=="6" goto BROWSER
if "%choice%"=="7" goto EXIT
goto MENU

:GUI
start "" "%~dp0VCM_Control_Center.exe"
goto MENU

:JUPYTER
echo Launching JupyterLab...
start "JupyterLab (8890)" cmd /c "Start-VCM-Notebook.bat"
goto MENU

:DEMO
echo Launching PC Assistant Live Demo...
start "Voice Assistant (7861)" cmd /c "Start-PC-Demo.bat"
goto MENU

:RECORDER
echo Launching Human Dataset Recorder...
start "Dataset Recorder (7862)" cmd /c "Start-VCM-Recorder.bat"
goto MENU

:STOP
echo Stopping services...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 7861,7862,8890 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"
echo Stopped all services on ports 7861, 7862, 8890.
timeout /t 2 >nul
goto MENU

:BROWSER
start http://127.0.0.1:8890/lab/tree/notebooks
start http://127.0.0.1:7861
start http://127.0.0.1:7862
goto MENU

:EXIT
exit
