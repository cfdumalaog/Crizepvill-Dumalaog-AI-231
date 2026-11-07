@echo off
setlocal
set "PROJECT=%USERPROFILE%\Desktop\MEng AI Notebooks\AI 222 231\AI 231\Dumalaog_ME3 - Voice Duplex\me3 qwen"
if not exist "%PROJECT%\scripts\Start-ME3-Pi-Web.ps1" (
  echo ME3 Qwen project was not found at:
  echo %PROJECT%
  pause
  exit /b 1
)
echo Tunneling the Pi UI, browser audio, and Spotify callback to this laptop.
echo First start ME3 Qwen Demo from the Raspberry Pi desktop.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PROJECT%\scripts\Start-ME3-Pi-Web.ps1" ^
  -SshTarget dalmacio@cfdfnjrpi5.local ^
  -HostKeyAlias 192.168.254.105 ^
  -RemoteProject /home/dalmacio/Desktop/dandan ^
  -LocalPort 8100 -LocalAudioPort 8101 ^
  -RemotePort 8126 -RemoteAudioPort 8127 ^
  -Brain remote -Variant qwen_instruct -OpenBrowser
if errorlevel 1 pause
