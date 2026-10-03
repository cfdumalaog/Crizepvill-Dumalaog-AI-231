@echo off
setlocal
set "APP=%~dp0ME2-VCM-Recorder\ME2 - VCM Recorder\ME2 - VCM Recorder.exe"
if not exist "%APP%" (
  echo The standalone recorder has not been built yet.
  echo Run scripts\build_desktop_recorder.ps1 from PowerShell.
  pause
  exit /b 1
)
start "ME2 - VCM on Raspberry Pi 5 — Recorder" "%APP%"
exit /b 0
