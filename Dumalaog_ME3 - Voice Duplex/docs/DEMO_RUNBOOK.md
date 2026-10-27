# ME3 demo runbook

This runbook is the shortest reliable path for demonstrating ME3 when the
Raspberry Pi microphone or speaker is unavailable. It keeps the real deployed
Qwen/tool loop and only replaces the physical audio endpoints with a WAV input
and a virtual discard sink.

## What is actually demonstrated

The injection demo exercises this sequence:

`voice fixture -> VAD -> ASR -> remote Qwen inference -> bounded ME3 tools -> spoken reply/TTS state -> evidence artifact`

The web-app scenario also demonstrates real file creation, file inspection,
web-server startup, and an HTTP response. No model training is involved.

This is strong agent/tool evidence, but it is not a claim that the physical
Fifine microphone or a physical speaker was exercised. Keep those evidence
lanes separate in the presentation and paper.

## Primary interaction: automatic full-duplex voice

For the actual ME3 interaction, do not use the browser Send button. Launch the
voice loop:

```powershell
& .\scripts\Start-Local-ME3.ps1 -Brain remote
```

The microphone remains open. Speak normally, stop speaking, and ME3 automatically
endpoints the turn, runs ASR, sends the transcript internally to Qwen, executes
approved tools when needed, and speaks the result. Speak again while it is
speaking to test barge-in; the current TTS playback is stopped and the new turn
is captured. The browser console at port 8124 is a secondary text/tool
diagnostic surface, not a replacement for this voice loop.

## Fastest Windows demo

The Windows workstation is the primary development and presentation target.
Run the workstation-first acceptance gate before promoting a new bundle to the
Pi:

```powershell
$py = '..\..\.venv\Scripts\python.exe'
& $py scripts\workstation_acceptance.py
```

The gate is local-only and does not send calls, SMS, email, or contact the Pi.
It writes `deployment/workstation_acceptance.json`; the Pi deployer refuses a
release whose acceptance record does not match the exact package hash. See
[`WORKSTATION_FIRST_RELEASE.md`](WORKSTATION_FIRST_RELEASE.md) for the full
promotion procedure.

From File Explorer, double-click:

`launchers\Start-Pi-Injection-Demo.bat`

The default `Status` scenario runs the deployed Pi agent with a system-status
fixture and an interrupt fixture. It prints a JSON result containing the tool
calls, virtual barge-in result, and artifact directory.

PowerShell equivalents from the ME3 project directory:

```powershell
# Status + virtual interruption
& .\scripts\Start-Pi-Injection-Demo.ps1 -Scenario Status

# Build a web app through Qwen tools, start its server, and verify it
& .\scripts\Start-Pi-Injection-Demo.ps1 -Scenario WebApp -OpenBrowser

# Inspect the generated SSH command without executing it
& .\scripts\Start-Pi-Injection-Demo.ps1 -Scenario WebApp -DryRun
```

The web-app result should include tools such as `write_file`, `list_files`,
`read_file`, `start_web_server`, and `web_servers`. The current Pi demo page is
served at `http://192.168.254.106:8090/` when the Pi server is running. The
Windows copy is served at `http://127.0.0.1:8123/` when the local demo server
is running.

For the ME2-style browser workflow, use the dedicated launchers:

```powershell
# Windows local preview; opens http://127.0.0.1:8123/
& .\scripts\Start-ME3-Web.ps1 -OpenBrowser

# Pi preview through an SSH loopback tunnel; opens http://127.0.0.1:8090/
& .\scripts\Start-ME3-Pi-Web.ps1 -StartServer -OpenBrowser
```

The Pi launcher serves only the bounded tracked `web/` presenter folder,
checks HTTP 200 before opening the tunnel, and keeps the endpoint private to
the SSH connection. Press `Ctrl+C` to close the tunnel. Neither launcher
publishes ME3 to the public Internet.

For the interactive workstation UI, which exercises the same ToolBrain without
opening physical audio or contacting Twilio/Resend, run:

```powershell
# localhost-only UI; choose Qwen + tools, Auto, or Offline stub in the page
& .\scripts\Start-ME3-UI.ps1 -OpenBrowser
```

Open `http://127.0.0.1:8124/`. Use **System status** first, then **Build a web
app** to demonstrate real `write_file`, verification, and bounded tool calls.
The UI returns a visible error if Qwen is unavailable; switch to **Offline
stub** to confirm the local browser and conversation path independently. The
the browser listener starts automatically after microphone permission; there is
no microphone button to press. It remains a browser convenience client and does
not replace the injected full ME3 audio evaluation below.

## What to show the professor

1. Run the `Status` scenario and point out `status: pass`, the Qwen tool names,
   and `barge_in_detected: true`.
2. Run the `WebApp` scenario and open the returned web-app URL. Point out that
   the files were created by tool calls rather than manually preloaded.
3. Open the generated artifact directory if the provenance of the run is
   requested. It contains the input mode, transcript/reply, tool-call names,
   virtual-output events, and safe audio topology.
4. State the hardware boundary explicitly: physical speaker playback and the
   paper-native FULL-DUPLEX-BENCH v1.5 capture protocol are not represented by
   an injection run.

## Physical Pi path

Before enabling the service, run the read-only gate on the Pi:

```bash
cd /home/dalmacio/Desktop/kiko/me3_voice_duplex
source .venv/bin/activate
python scripts/pi_audio_preflight.py --json
systemctl --user is-enabled me3-agent
systemctl --user is-active me3-agent
```

Only enable `me3-agent` when the Fifine input has a positive PortAudio input
channel count and a known playback route is available. During the 2026-10-26
readiness check, ALSA listed the Fifine capture card, but PortAudio reported
zero available input channels because the existing protected VCM process was
using the device. ME3 was therefore left disabled/inactive, with no restart
loop. Do not stop that VCM process as part of this demo runbook; use a second
capture device or an explicitly authorized maintenance window.

The split-device acceptance path remains available when the Pi can release the
Fifine input:

```powershell
& '..\..\.venv\Scripts\python.exe' -m me3.agent --brain remote `
    --ssh-mic 'dalmacio@cfdfnjrpi5.local' --remote-device 'hw:2,0'
```

It captures from the Pi and plays TTS through the Windows workstation speakers.

## Readiness checklist

- Local tests: `pytest tests -q`.
- Local dependencies: `scripts/dev_env_check.py`.
- Four MCP health tools plus PortAudio topology:
  `scripts/local_smoke_test.py --with-audio`.
- Source/paper contract: `scripts/check_paper_alignment.py`.
- Pi model/tool health: `python -m me3.agent --check --load-models`.
- Pi injection status and web-app scenarios: the launcher above.
- Browser reachability: HTTP 200 from the Pi page and local page.
- Physical mic/speaker: only claim after `pi_audio_preflight.py` passes and a
  human turn is recorded with the hardware owner/process state documented.
- ElevenLabs Filipino voice: only claim live use after an authorized
  provider call; otherwise `tts.engine: auto` uses the local Piper fallback.
- Call/SMS/email tools: run `channel_health` for a network-free configuration
  check. Do not set `confirm=true` in a classroom demo; live Twilio/Resend
  tests require an operator-owned destination and an explicit out-of-band
  approval. See `docs/CHANNEL_INTEGRATIONS.md` for the required APIs and the
  remaining TwiML/media-stream, consent, and human-handoff gates.
