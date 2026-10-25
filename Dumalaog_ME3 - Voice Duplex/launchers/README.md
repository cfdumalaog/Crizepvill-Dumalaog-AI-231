# ME3 quick launch

Double-click these Windows launchers from File Explorer:

| Launcher | What it opens | Notes |
|---|---|---|
| `Start-ME3.bat` | The full-duplex voice agent (console app) | No browser/ports. The console stays open while the agent listens; speak to the microphone and interrupt it mid-sentence to barge in. `Ctrl+C` (or closing the window) stops it. |
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
