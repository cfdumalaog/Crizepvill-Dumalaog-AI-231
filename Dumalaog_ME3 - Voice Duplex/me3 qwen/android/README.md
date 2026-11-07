# ME3 Qwen Android companion

This is a small native Android client for the active ME3 Qwen Instruct Pi
service. It packages the existing Assistant UI in a WebView with Android
permission handling and a remembered editable server address.

The APK does **not** move the Python voice loop or Qwen models onto the phone.
The Raspberry Pi remains the ME3 runtime: it owns the Fifine microphone, VAD,
ASR/TTS providers, Qwen ToolBrain, bounded tools, and the HTTP/WebSocket
server. The Android device is an operator console and browser PCM speaker over
the trusted LAN. This is the same split as the existing Pi browser workflow.

## Build

From the workspace root, use the existing local Gradle distribution and Android
SDK:

```powershell
& '.\Budget App\tools\gradle-8.11.1\bin\gradle.bat' `
  -p '.\AI 231\Dumalaog_ME3 - Voice Duplex\me3 qwen\android' `
  assembleDebug --no-daemon
```

The debug APK is written to:

```text
app\build\outputs\apk\debug\app-debug.apk
```

It is also copied to the Android project root as
`ME3-Qwen-Android-debug.apk` after a successful build. Install it with:

```powershell
adb install -r '.\AI 231\Dumalaog_ME3 - Voice Duplex\me3 qwen\android\ME3-Qwen-Android-debug.apk'
```

No API key, Qwen credential, VPN profile, or Pi secret is bundled. The default
address is the currently documented Pi UI address, but the first screen lets
the operator edit it and the last successful address is remembered locally.

## Use

1. Put the Android phone and Pi on the same trusted LAN or approved VPN.
2. Enter the Pi ME3 HTTP address, normally `http://<PI-IP>:8126/`.
3. Tap **Connect to ME3**.
4. In the ME3 Assistant page, tap **Connect browser speaker** once so Android
   allows Web Audio playback.
5. Keep only one ME3 browser/audio client connected to preserve the Pi echo
   reference. The Pi microphone remains the authoritative input device.

The app permits cleartext HTTP because the current Pi service uses HTTP. It is
not a public-hosting or transport-security feature; keep the service behind
the trusted LAN/VPN boundary described by the ME3 runbook.
