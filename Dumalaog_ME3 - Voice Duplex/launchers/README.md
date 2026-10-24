# ME3 quick launch

Double-click these Windows launchers from File Explorer:

| Launcher | What it opens | Notes |
|---|---|---|
| `Start-ME3.bat` | The full-duplex voice agent (console app) | No browser/ports. The console stays open while the agent listens; speak to the microphone and interrupt it mid-sentence to barge in. `Ctrl+C` (or closing the window) stops it. |

It uses the single shared workspace `.venv` (same as the ME2 launchers).
The LLM brain is the remote vLLM endpoint from `config/config.yaml` when
`VLLM_API_KEY` is set in the environment; otherwise it runs with the stub
brain so the whole voice pipeline (VAD -> ASR -> LLM -> TTS) still works
end-to-end. First run downloads the Piper voice model into `models/tts/`.

PowerShell overrides:

```powershell
& '.\scripts\Start-Local-ME3.ps1' -Device 'Fifine'     # pick a mic
& '.\scripts\Start-Local-ME3.ps1' -Brain stub          # force canned replies
```

Verification without the microphone:

```powershell
& '..\..\.venv\Scripts\python.exe' -m me3.agent --check --load-models
```
