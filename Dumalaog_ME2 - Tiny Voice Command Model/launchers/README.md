# ME2 quick launch

Double-click these Windows launchers from File Explorer:

| Launcher | What it opens | Address / notes |
|---|---|---|
| `Start-VCM.bat` | Local two-stage wake + intent demo | Open `http://127.0.0.1:7863/studio`. The console stays open while the server runs. |
| `ME2-VCM-Recorder\ME2 - VCM Recorder\ME2 - VCM Recorder.exe` | Native human speech dataset recorder | Direct Windows audio capture; no browser or web server. Start â†’ Stop â†’ review live spectrogram â†’ Save. |
| `Start-Recorder-App.bat` | Starts the compiled native recorder | Use this if you prefer a launcher instead of opening the EXE directly. |
| `Start-Recorder.bat` | Browser recorder for development | Opens `http://127.0.0.1:7862`; browser microphone permission required. If a recorder is already running, the launcher detects it and opens the existing page instead of failing; otherwise it starts the server in its own console window and opens the browser once it answers on 7862. Do not run it alongside the native recorder. |
| `Open-Notebooks.bat` | JupyterLab notebook browser | Opens the `notebooks/` folder on port 8890. Start with the two active notebooks listed in `notebooks/README.md`. |

Port 7861 is retired. Port 7864 belongs to the separate Desktop launcher that
creates an SSH tunnel to the default Pi VCM; it is not the Windows-local VCM.
The isolated Pi candidate uses port 7865 and remains a separate trial.

The VCM, notebook and browser-development launcher use the single shared workspace `.venv`. The compiled recorder does not need Python after it is built. Close an older VCM process before starting another on
port 7863; the VCM launcher will report when that port is occupied rather than
silently starting a second microphone listener. You can override the port from
a PowerShell prompt, for example `& '.\scripts\Start-Local-VCM.ps1' -Port 7864`.

The VCM's Assistant and Studio pages include a restart button. In Studio, choose
the microphone first, then restart to switch the live listener. The speaker
dropdown routes music/page media on browsers that support audio output
selection; browser text-to-speech remains on the system default output. The
native recorder has microphone/speaker dropdowns, explicit Start/Stop/Play/Save controls, and a live spectrogram. It uses the same labels, phrase variants, speaker/condition fields, consent requirement, WAV/CSV/Excel saving, and restart behavior. Build with `scripts\build_desktop_recorder.ps1`. Save any pending take before restarting. Existing browser-recorder processes must be restarted after source edits.

The binary wake model is still an experimental candidate: the app can run, but
live multi-speaker wake accuracy has not been validated.
