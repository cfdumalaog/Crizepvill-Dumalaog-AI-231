@echo off
setlocal
for %%I in ("%~dp0..") do set "PROJECT=%%~fI"
echo Starting ME3 FASTER-WHISPER + PIPER from the Pi at 192.168.254.105...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PROJECT%\scripts\Start-ME3-Pi-Web.ps1" ^
  -SshTarget dalmacio@192.168.254.105 ^
  -RemoteProject /home/dalmacio/Desktop/dandan ^
  -LocalPort 8090 -LocalAudioPort 8091 ^
  -RemotePort 8126 -RemoteAudioPort 8127 ^
  -Brain remote -Variant current -StartServer -OpenBrowser
if errorlevel 1 pause
