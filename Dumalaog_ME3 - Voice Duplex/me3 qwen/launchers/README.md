# ME3 quick launch

Double-click these Windows launchers from File Explorer:

| Launcher | What it opens | Notes |
|---|---|---|
| `Start-ME3.bat` | **Primary full-duplex voice agent** | No Send button: the mic stays live, VAD endpoints turns, Qwen or GPT-5.6 Luna tools execute automatically, TTS speaks, and new speech barges in. `Ctrl+C` stops it. |
| `Start-ME3-Qwen-Instruct.bat` | Qwen Instruct audio variant | Selects Qwen2-Audio ASR and Qwen3-TTS while keeping the same text ToolBrain, tools, and barge-in path. Reads the protected handoff in memory. |
| `Start-ME3-UI.bat` | Interactive local agent console | Opens `http://127.0.0.1:8124/`; Qwen + tools, GPT-5.6 Luna + tools, or offline stub, tool events, and hard-disabled Telegram/Resend side effects. |
| `Start-ME3-Web.bat` | Local browser preview | Serves the tracked `web\` presenter surface on `http://127.0.0.1:8123/` and reports LAN URLs. Pass `-Path agent_workspace\voice_webapp` for a generated app. |
| `Start-ME3-Pi-Web.bat` | ME2-style Pi browser tunnel | Starts the Pi static preview when available, forwards it to `http://127.0.0.1:8090/`, and opens the browser. `Ctrl+C` closes the tunnel. |
| `Start-ME3-Pi-Qwen.bat` | **Named Qwen Instruct laptop launcher** | Connects to the Pi by stable mDNS name (`cfdfnjrpi5.local`) and forwards Pi ports 8126/8127 to laptop ports 8100/8101, including the Spotify OAuth callback on port 8100. It does not start a second ME3 server. |
| `pi/ME3-Qwen-Demo.desktop` | **Pi desktop launcher** | Restarts/enables the Pi's `me3-agent.service`, checks the web/audio service and voice loop, and displays current Pi addresses for the laptop launcher. |
| `Start-Pi-Injection-Demo.bat` | The deployed Pi Qwen/tool demo | Uses reviewed WAV fixtures and a virtual sink; recommended when the physical Fifine input is occupied or the speaker path is not ready. |

It uses the single shared workspace `.venv` (same as the ME2 launchers).
The LLM brain is Qwen/vLLM when `VLLM_API_KEY` is set, GPT-5.6 Luna when
`OPENAI_API_KEY` is set and Qwen is unavailable, or the stub otherwise. The
whole voice pipeline (VAD -> ASR -> LLM -> TTS) remains the same. First run
downloads the Piper voice model into `models/tts/`.

PowerShell overrides:

```powershell
& '.\scripts\Start-Local-ME3.ps1' -Device 'Fifine'     # pick a mic
& '.\scripts\Start-Local-ME3.ps1' -Brain luna         # GPT-5.6 Luna medium
& '.\scripts\Start-Local-ME3.ps1' -Brain stub          # force canned replies
& '.\scripts\Start-Local-ME3.ps1' -Brain remote -Variant qwen_instruct
# Pi Fifine microphone -> this workstation's default speakers
& '..\..\..\.venv\Scripts\python.exe' scripts\run_me3_module.py me3.agent --brain remote `
    --ssh-mic 'dalmacio@cfdfnjrpi5.local' --remote-device 'hw:2,0'
```

The split-device command is a local acceptance test. It uses SSH only for
raw microphone PCM and never prints or transfers the Qwen credential; TTS is
played by the workstation's normal speaker device. The Pi production command
(`python scripts/run_me3_module.py me3.agent`) remains host-local and uses the Pi's own capture and
playback devices.

Verification without the microphone:

```powershell
& '..\..\..\.venv\Scripts\python.exe' scripts\run_me3_module.py me3.agent --check --load-models
```

For the full presentation sequence, including the web-app scenario and the
physical-audio boundary, see [`docs/DEMO_RUNBOOK.md`](../docs/DEMO_RUNBOOK.md).

## Version hygiene

These seven launchers are intentionally different role-based entry points, not
old copies of one launcher. The current release is the only bundle left in
`dist/`; historical release packages (including their older launcher copies),
old PowerPoint versions, and stale generated build outputs are recoverable
under [`archive/README.md`](../archive/README.md).

## Interactive local UI (diagnostic console)

The UI is a workstation diagnostic console, not the primary voice interaction.
The primary path is `Start-ME3.bat`: it listens continuously and automatically
passes each VAD-ended utterance to the same ToolBrain and bounded project-tool
dispatcher, then speaks the response. There is no manual Send step in that
path. The browser UI calls the same brain for explicit text/tool inspection:

```powershell
& '.\scripts\Start-ME3-UI.ps1' -OpenBrowser       # auto: Qwen when configured, else stub
& '.\scripts\Start-ME3-UI.ps1' -Brain qwen        # require the Qwen endpoint
& '.\scripts\Start-ME3-UI.ps1' -Brain luna        # GPT-5.6 Luna, reasoning disabled
& '.\scripts\Start-ME3-UI.ps1' -Brain stub        # fully offline conversation check
```

The page includes starter prompts for system status, channel readiness, project
inspection, and real web-app creation. Its browser listener starts automatically
after microphone permission and submits detected utterances without a voice
button; the authoritative ME3 VAD -> Whisper -> brain -> TTS path remains the
full-duplex launcher and injection evaluator. Telegram and email are
hard-disabled in this local UI even if provider credentials exist.

While a request is running, the page exposes a `Show task progress (not model reasoning)` toggle.
When enabled, it shows the current task and a live, safe progress summary such
as model inference, tool planning, `write_file`/`run_command` execution, and
completion. It intentionally does not expose private model chain-of-thought or
credentials. The panel uses the asynchronous `/api/chat/start` and
`/api/chat/status` endpoints; the original synchronous `/api/chat` endpoint is
kept for compatibility.

## Browser access

For a Windows-local ME3 web app, double-click `Start-ME3-Web.bat` or run:

```powershell
& '.\scripts\Start-ME3-Web.ps1' -OpenBrowser
```

The server binds to `0.0.0.0` so the local URL and machine LAN URLs can be
used. This is local-network access only; it does not publish the app to the
Internet. Stop it with `Ctrl+C`.

For the Pi equivalent, double-click `Start-ME3-Pi-Web.bat`. It starts the
bounded `web` presenter surface on the Pi, checks HTTP 200, and
creates an SSH loopback tunnel at:

```text
http://127.0.0.1:8090/
```

The tunnel is the ME3 equivalent of the ME2 local browser tunnel. It is not a
public URL. To forward an already-running Pi server without starting one:

```powershell
& '.\scripts\Start-ME3-Pi-Web.ps1' -OpenBrowser
```

Use `-StopServer` only for a server started by this launcher.

For the named Qwen launcher, double-click `Start-ME3-Pi-Qwen.bat`. It selects
the `ME3 QWEN INSTRUCT` profile explicitly. The Pi reads the protected
`QWEN_INSTRUCT_ASR_BASE_URL`, `QWEN_INSTRUCT_TTS_BASE_URL`, and
`QWEN_INSTRUCT_API_KEY` values from the dedicated mode-600
`~/.config/me3/qwen-instruct.env`; the ordinary installer cannot overwrite
this file during future upgrades.

The named Qwen launcher forwards only the Pi UI and browser-audio ports by
default. The Pi's SHARC VPN supplies direct access to
`10.153.60.233:8000/8001`, so the laptop does not need to remain connected to
the VPN for model traffic. The legacy model reverse tunnel is still available
only through `Start-ME3-Pi-Web.ps1 -UseReverseTunnel` when the Pi VPN is
intentionally unavailable. Because this service owns the physical Fifine
microphone, run only this one full-duplex Qwen listener at a time.
