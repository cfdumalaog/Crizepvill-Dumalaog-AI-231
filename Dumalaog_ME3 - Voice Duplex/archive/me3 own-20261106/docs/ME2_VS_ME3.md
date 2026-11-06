# ME2 versus ME3

## Short version

ME2 is a compact, offline, closed-vocabulary command appliance. It listens
for a wake word and then maps a short utterance to one of 31 fixed intent
labels using two scratch-trained TinyDSCNN-48 ONNX classifiers. The action code
is known before runtime: play a bundled melody, change a simulated light,
report weather, set a timer, and similar demonstrations.

ME3 keeps the useful ME2 operating pattern—Raspberry Pi service, selectable
audio devices, a browser dashboard, device status, and a reproducible
benchmark—but replaces the fixed classifier/action table with a continuous
full-duplex agent loop:

```text
ME2:  microphone -> wake classifier -> intent classifier -> fixed action
ME3:  Pi microphone -> Silero VAD -> ASR -> Qwen/Luna tool reasoning
      -> MCP/local tools -> TTS -> Pi speaker or browser speaker
                    ^ continuous capture + barge-in while speaking ^
```

No ME3 model training is required. Qwen/Luna and the optional ElevenLabs or
Qwen audio services are inference APIs; ME3 supplies the orchestration,
MCP servers, VAD policy, tool safety gates, device transport, benchmark
adapter, and deployment package.

## Side-by-side engineering differences

| Area | ME2 | ME3 |
|---|---|---|
| Runtime model | Two trained TinyDSCNN-48 ONNX classifiers | API/local ASR + VAD + provider-neutral tool-calling LLM + TTS |
| Language | Dataset labels and fixed phrase variation | Filipino/English/Taglish ASR and open-ended requests |
| Interaction | Wake, listen, classify, execute one coded intent | Always listening, endpointing, thinking, speaking, and interruptible barge-in |
| Actions | Code paths for a finite 31-label set | MCP/tool registry for files, commands, web apps, research, timers, system status, and gated channels |
| Voice | Bundled response WAVs / local output | Piper, ElevenLabs Filipino-capable voice, or Qwen3-TTS |
| Network | Offline-first; weather is a bounded demo API | Qwen/Luna/ElevenLabs are optional remote services; local stub/injection lanes remain available |
| Browser | ME2 dashboard and recorder controls | ME2-style live dashboard plus thinking/task progress, model/tool timing, variant, routing, and browser PCM speaker |
| Audio topology | Pi microphone and Pi output device | Pi microphone stays on Pi; output is selectable as Pi speaker or connected PC browser speaker over LAN/SSH tunnel |
| Evaluation | Static/live classifier accuracy and inference latency | Full-duplex turn-taking, missed/false turn-taking, interruption, stop/response latency, and v1.5 behavior categories |
| Deployment | Copy model pair and launcher to Pi | Workstation-accepted release with both ME3 variants, MCP services, UI/audio server, and vendored fd-benchmark adapter |

## What remains the same as ME2

The user-facing workflow is intentionally familiar: start one Pi service,
open the web page from a PC on the same network, inspect a live status surface,
choose the microphone/output route, and run the agent without a browser
``Send`` button. The Pi still owns the physical microphone. If browser output
is selected, the page must be connected once by clicking **Connect browser
speaker** because browsers prohibit unsolicited audio autoplay.

## What is not equivalent

ME3 is not a drop-in replacement for ME2's 31-class accuracy claim. An
open-ended tool-calling agent has different failure modes: ASR errors,
provider/network failures, hallucinated or unsafe actions, tool authorization,
latency, endpointing, and interruption behavior. Therefore ME3 reports
scenario-level and turn-taking metrics rather than treating every utterance as
one classifier label. ME2's model-size/accuracy and Pi CPU results remain valid
for ME2 only.
