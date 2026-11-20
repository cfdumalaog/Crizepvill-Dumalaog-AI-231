# ME3 Qwen Instruct - Two-Minute Presentation Script

**Estimated delivery time:** approximately 2 minutes at a clear presentation pace.

## Slide 1 - System, prompt, and context

ME3 Qwen Instruct is a full-duplex voice agent on a Raspberry Pi. The Fifine microphone is captured at 16-kilohertz mono. Echo guard and Silero VAD detect speech, reject self-echo, and keep capture open while TTS plays. User overlap is queued; stop, skip, and steer are explicit controls.

Speech flows through Qwen2-Audio ASR, the Qwen3.6-27B ToolBrain, four owned MCP servers with 54 tools, and Qwen3-TTS 1.7B using Aiden. Sixteen-kilohertz PCM travels through a browser WebSocket to the laptop speaker. The Pi owns VAD, echo, safety, and duplex state; Qwen services are reached over VPN. No training is required.

The prompt defines a Philippines customer-support assistant for Filipino, English, and Taglish. It requires formal, concise speech, Pi-user tools, no credentials, and no unsupported claims. For long work it says, "I am processing your request now." Context management keeps fast mode and hidden thinking off, uses 24 tool rounds for ordinary turns and up to 48 for software tasks, retains short history and tool results, queues the latest prompt, reports background status, and exposes only sanitized terminal information.

## Slide 2 - MCP services, benchmark, and results

Five roles are covered: ToolBrain handles MCP function calling; ASR transcribes; TTS speaks; VAD performs Pi-CPU endpointing with Silero ONNX; and task/channel tools handle files, web, builds, timers, ME2 parity, and channels. Lights and thermostat actions are simulated; external APIs need authorization.

The official direct-feed benchmark uses one clock and injected audio only: no physical microphone or speaker, acoustic feedback, tools, fillers, backchannels, or channel effects. All 40 of 40 trials completed: 30 stage-latency trials and 10 group clips, with no judge. Scoring used faster-whisper/small.

The measured mean and p95 values were E2E 4.89/7.01 seconds, target 4.0; ASR 1.17/2.16, target 0.8; LLM first token 1.35/4.73, target 1.5; TTS first audio 2.51/4.28, target 0.6; barge-in 0.19/0.19, target 0.8; and WER 24.1 percent, target 15 percent. Barge-in passed. LLM mean passed, but p95 missed; all other gates missed.

Remaining challenges are Spotify API and PKCE, Calendar OAuth, Telegram setup, Android call permissions, and replacing simulated devices with authorized APIs.
