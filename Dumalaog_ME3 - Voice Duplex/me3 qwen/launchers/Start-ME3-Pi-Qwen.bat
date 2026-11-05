@echo off
setlocal
for %%I in ("%~dp0..") do set "PROJECT=%%~fI"
echo Starting ME3 QWEN INSTRUCT from the Pi at 192.168.254.105...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PROJECT%\scripts\Start-ME3-Pi-Web.ps1" ^
  -SshTarget dalmacio@192.168.254.105 ^
  -RemoteProject /home/dalmacio/Desktop/dandan-qwen ^
  -LocalPort 8100 -LocalAudioPort 8101 ^
  -RemotePort 8136 -RemoteAudioPort 8137 ^
  -Brain remote -Variant qwen_instruct -StartServer -OpenBrowser
if errorlevel 1 pause
