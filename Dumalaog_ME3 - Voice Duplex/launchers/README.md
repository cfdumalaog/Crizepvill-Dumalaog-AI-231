# ME3 quick launch

Double-click these Windows launchers from File Explorer:

| Launcher | What it opens | Notes |
|---|---|---|
| `Start-ME3.bat` | The full-duplex voice agent (console app) | No browser/ports. The console stays open while the agent listens; speak to the microphone and interrupt it mid-sentence to barge in. `Ctrl+C` (or closing the window) stops it. |
| `Start-ME3-Web.bat` | Local browser preview | Serves the tracked `web\` presenter surface on `http://127.0.0.1:8123/` and reports LAN URLs. Pass `-Path agent_workspace\voice_webapp` for a generated app. |
| `Start-ME3-Pi-Web.bat` | ME2-style Pi browser tunnel | Starts the Pi static preview when available, forwards it to `http://127.0.0.1:8090/`, and opens the browser. `Ctrl+C` closes the tunnel. |
| `Start-Pi-Injection-Demo.bat` | The deployed Pi Qwen/tool demo | Uses reviewed WAV fixtures and a virtual sink; recommended when the physical Fifine input is occupied or the speaker path is not ready. |

It uses the single shared workspace `.venv` (same as the ME2 launchers).
The LLM brain is the remote vLLM endpoint from `config/config.yaml` when
`VLLM_API_KEY` is set in the environment; otherwise it runs with the stub
brain so the whole voice pipeline (VAD -> ASR -> LLM -> TTS) still works
end-to-end. First run downloads the Piper voice model into `models/tts/`.

PowerShell overrides:

```powershell
& '.\scripts\Start-Local-ME3.ps1' -Device 'Fifine'     # pick a mic
& '.\scripts\Start-Local-ME3.ps1' -Brain stub          # force canned replies
# Pi Fifine microphone -> this workstation's default speakers
& '..\..\.venv\Scripts\python.exe' -m me3.agent --brain remote `
    --ssh-mic 'dalmacio@cfdfnjrpi5.local' --remote-device 'hw:2,0'
```

The split-device command is a local acceptance test. It uses SSH only for
raw microphone PCM and never prints or transfers the Qwen credential; TTS is
played by the workstation's normal speaker device. The Pi production command
(`python -m me3.agent`) remains host-local and uses the Pi's own capture and
playback devices.

Verification without the microphone:

```powershell
& '..\..\.venv\Scripts\python.exe' -m me3.agent --check --load-models
```

For the full presentation sequence, including the web-app scenario and the
physical-audio boundary, see [`docs/DEMO_RUNBOOK.md`](../docs/DEMO_RUNBOOK.md).

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
