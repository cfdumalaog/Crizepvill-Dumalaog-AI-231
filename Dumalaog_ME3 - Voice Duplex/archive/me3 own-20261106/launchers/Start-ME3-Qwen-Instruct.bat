@echo off
setlocal
for %%I in ("%~dp0..") do set "PROJECT=%%~fI"
if not exist "%PROJECT%\scripts\Start-Local-ME3.ps1" (
  echo Could not find the ME3 launcher under "%PROJECT%\scripts".
  pause
  exit /b 1
)
echo Starting ME3 with the Qwen Instruct ASR/TTS variant.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PROJECT%\scripts\Start-Local-ME3.ps1" -Variant qwen_instruct %*
if errorlevel 1 pause
