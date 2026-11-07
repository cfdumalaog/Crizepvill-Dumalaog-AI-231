@echo off
setlocal
for %%I in ("%~dp0..") do set "PROJECT=%%~fI"
echo Connecting to the canonical ME3 QWEN INSTRUCT systemd service on the Pi...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PROJECT%\scripts\Start-ME3-Pi-Web.ps1" ^
  -SshTarget dalmacio@cfdfnjrpi5.local ^
  -HostKeyAlias 192.168.254.105 ^
  -RemoteProject /home/dalmacio/Desktop/dandan ^
  -LocalPort 8100 -LocalAudioPort 8101 ^
  -RemotePort 8126 -RemoteAudioPort 8127 ^
  -Brain remote -Variant qwen_instruct -OpenBrowser
if errorlevel 1 pause
