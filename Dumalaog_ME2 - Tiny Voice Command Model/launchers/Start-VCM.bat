@echo off
setlocal
for %%I in ("%~dp0..") do set "PROJECT=%%~fI"
if not exist "%PROJECT%\scripts\Start-Local-VCM.ps1" (
  echo Could not find the VCM launcher under "%PROJECT%\scripts".
  pause
  exit /b 1
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PROJECT%\scripts\Start-Local-VCM.ps1" %*
if errorlevel 1 pause
