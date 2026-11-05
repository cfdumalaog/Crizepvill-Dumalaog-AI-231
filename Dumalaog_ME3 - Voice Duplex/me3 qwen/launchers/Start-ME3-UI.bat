@echo off
setlocal
for %%I in ("%~dp0..") do set "PROJECT=%%~fI"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PROJECT%\scripts\Start-ME3-UI.ps1" %*
if errorlevel 1 pause
