# ME3 Qwen Android companion

This is the native Android APK for the active ME3 Qwen Instruct Pi service. It
packages the deployed Assistant UI in a WebView with Android permission
handling and a remembered editable server address. The operator surface is
deliberately focused: the animated ME3 sound logo shows listening/speaking
state, and the chatbox is reserved for the conversation, approvals, and private
provider setup prompts.

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

## Provider/model seed and credential boundary

The APK is seeded with the deployed runtime identity through the Pi page:

- provider stack: Qwen Instruct;
- text model: `Qwen/Qwen3.6-27B`;
- ASR model: `Qwen/Qwen2-Audio-7B-Instruct`;
- TTS model/voice: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` / `aiden`;
- default Pi service: `http://cfdfnjrpi5.local:8126/` (editable for a changed
  DHCP address).

Provider credentials are intentionally **not embedded in the APK**. An APK is
extractable, so encrypting a key before putting it in the binary would not make
it secret: the installed app would also contain the decryption path. The
deployed Pi already stores its configured credentials in the protected private
runtime and the APK receives only configured/not-configured status. OpenAI ASR
and ElevenLabs TTS are not configured in the current deployed runtime, so they
cannot truthfully be seeded as working providers. Any optional credential setup
must be entered through ME3's private setup flow and never committed or logged.

## Planned parity work

The detailed staged plan is in
[`docs/APK_MIGRATION_PLAN.md`](../docs/APK_MIGRATION_PLAN.md). The target is
voice-first Codex-like parity: one conversation for voice and typed requests,
live task/tool progress while a job runs, model-layer dropdowns, private setup
prompts, integration actions, artifacts, and restart/reconnect controls.

Provider keys are intentionally not seeded into this APK. The normal mode keeps
all credentials on the Pi's private store and sends only redacted status to the
client. If a future direct-phone mode is required, it must use Android Keystore
and app-private storage with rotation; putting a key in Java/Kotlin, resources,
WebView storage, or the APK would make it extractable.
