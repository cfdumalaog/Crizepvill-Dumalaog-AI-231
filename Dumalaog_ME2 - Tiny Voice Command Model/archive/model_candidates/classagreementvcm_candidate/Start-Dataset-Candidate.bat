@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Dataset-Candidate.ps1" %*
endlocal
