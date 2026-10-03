@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Sync-VCM.ps1" %*
if errorlevel 1 pause
