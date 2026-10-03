@echo off
title Tiny VCM - Voice Assistant Live Demo (7861)
cd /d "%~dp0"
echo ==============================================================================
echo     TINY VOICE COMMAND MODEL (VCM) - LIVE PC ASSISTANT DEMO
echo ==============================================================================
echo Continuous Microphone Stream on Port 7861...
echo Say "Hi Dandan" or "Hello Dandan" into your room to awaken the assistant!
echo URL: http://127.0.0.1:7861
echo Press Ctrl+C to stop.
echo ==============================================================================
".venv\Scripts\python.exe" "AI 231\Dumalaog_ME2 - Tiny Voice Command Model\demo.py"
pause
