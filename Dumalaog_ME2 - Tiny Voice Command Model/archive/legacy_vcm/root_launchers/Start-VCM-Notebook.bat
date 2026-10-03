@echo off
title JupyterLab Notebook Server (8890)
cd /d "%~dp0"
echo Starting JupyterLab Notebook Server on port 8890...
".venv\Scripts\python.exe" "AI 231\Dumalaog_ME2 - Tiny Voice Command Model\scripts\start_jupyter.py"
pause
