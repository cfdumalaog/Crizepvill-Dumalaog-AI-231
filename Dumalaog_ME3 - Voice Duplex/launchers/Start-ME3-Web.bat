@echo off
setlocal
for %%I in ("%~dp0..") do set "PROJECT=%%~fI"
if not exist "%PROJECT%\scripts\Start-ME3-Web.ps1" (
  echo Could not find the ME3 local web launcher under "%PROJECT%\scripts".
  pause
  exit /b 1
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PROJECT%\scripts\Start-ME3-Web.ps1" -OpenBrowser %*
if errorlevel 1 pause
