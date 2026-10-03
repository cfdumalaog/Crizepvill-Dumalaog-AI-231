@echo off
title Tiny VCM - Human Dataset Recorder UI (7862)
cd /d "%~dp0"
echo ==============================================================================
echo     TINY VOICE COMMAND MODEL (VCM) - DATASET RECORDER UI
echo ==============================================================================
echo Opening Human Voice Recorder on Port 7862...
echo NOTE: Make sure the Voice Assistant Demo is muted/closed while recording!
echo URL: http://127.0.0.1:7862
echo Press Ctrl+C to stop.
echo ==============================================================================
".venv\Scripts\python.exe" "AI 231\Dumalaog_ME2 - Tiny Voice Command Model\record_dataset.py"
pause
