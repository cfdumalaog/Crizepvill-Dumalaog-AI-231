# ME3 Qwen benchmark and Pi audio evidence — 20 November 2026

## Current release boundary

The supported demo line is **ME3 Qwen Instruct**. The live Pi service is the
Qwen audio composition with Silero VAD and the Gemini ToolBrain. The service
does not train a model: it orchestrates the selected inference endpoints,
MCP-owned actions, duplex state, echo protection, and verification.

The current Pi service was verified healthy after the audio-default update:

- Input: `fifine Microphone` (Fifine USB microphone).
- Output: `USB Audio` (Pi USB speaker path).
- Service: `me3-agent.service`, HTTP 8126 and audio WebSocket 8127.
- Health: `/api/health` returned `ok=true`, `loop=true`.
- Active audio profile: Qwen2-Audio ASR, Qwen3-TTS CustomVoice `aiden`, and
  local Silero VAD.

## Class benchmark protocol

The supplied fdbench class protocol is **direct-feed/injection-only**. It uses
one benchmark clock and injects the supplied WAV clips into the assistant. It
does not open the physical Pi microphone or speaker. Tools, agent calls,
fillers, and backchannels are disabled for the class latency measurement. A
full round is 10 phrase groups; the standard run is 10 phrases × 3 repeated
stage runs plus one full-duplex round: 40 planned trials.

WER is a corpus metric, not a percentile. It is calculated from the reference
words and final ASR transcript using aggregate word edits:

```text
WER = total insertions + deletions + substitutions
      -------------------------------------------------
                  total reference words
```

The target source of truth is the vendored fd-benchmark target file:
E2E 4.0 s, ASR final 0.8 s, LLM TTFT 1.5 s, TTS TTFB 0.6 s, barge-in 0.8 s,
and corpus WER 15%. A latency row is `met` only when both its mean and p95 are
under target. WER is judged against its aggregate corpus ratio and has no p95.
For barge-in, the benchmark measures `t_stop`: the end of the assistant
segment that was playing at overlap onset minus the overlap onset. A trial in
which the assistant talks through the overlap is counted separately, not as a
latency sample.

## Completed full direct-feed run

The latest completed run is retained at:

`benchmarking/fd-benchmark/runs/ME3-QWEN/2026-11-19_02-24-57/`

It completed 40/40 planned trials and produced `results.json`, `results.csv`,
`results.html`, and `summary.md`. The result is a deployed-pipeline fallback
diagnostic, not a clean remote-Qwen score, because DGX3 ports 8000/8001 timed
out during the run and the Pi used the cached faster-whisper/small fallback.

| Metric | Mean | p95 | n / detail | Target |
|---|---:|---:|---:|---:|
| End-to-end | 4.30 s | 7.19 s | 28 | 4.0 s |
| ASR final | 2.68 s | 5.32 s | 29 | 0.8 s |
| LLM TTFT | — | — | 0 stage marks | 1.5 s |
| TTS TTFB | 0.63 s | 0.62 s | 29 | 0.6 s |
| Barge-in | — | — | 0 valid samples | 0.8 s |
| **Corpus WER** | **17.6%** | **not applicable** | **31 edits / 176 reference words; n=29** | **15%** |

The run’s mean phrase WER was 13.25%; the reported class WER is the aggregate
corpus value, 31/176 = 17.61%, and is the value used in the deck.

## Pi physical acoustic gate

The acoustic check is deliberately separate from fdbench. A known speech WAV
fixture was played through the Pi USB speaker while ME3 captured the Fifine
microphone. The service remained on the Pi audio path. The observed state
reported:

- `selected_input=fifine Microphone`.
- `selected_output=USB Audio`.
- 4,429 echo-cancelled frames.
- 375 echo-suspect frames.
- Peak playback/capture correlation: 0.807.
- 0 accepted self-transcripts and 0 runtime errors during the fixture check.

This is evidence that the playback reference reached the capture path and was
handled by the echo guard. It is **not** a human barge-in, WER, or full v1.5
turn-taking result. The remaining physical gate is one clean human sentence
overlapping a live Pi speaker response, with the barge-in latency measured.

## 20 November rerun diagnostic

Two fresh full-run attempts were started on the Pi after adding a benchmark-only
switch that skips unrelated network backups. Both were stopped when the
assistant stopped emitting events after five no-reply trials; they are retained
as diagnostic folders and are not included in the score:

- `benchmarking/fd-benchmark/runs/ME3-QWEN-GEMINI-20261120/2026-11-20_08-47-04/`
- `benchmarking/fd-benchmark/runs/ME3-QWEN-AUDIO-20261120/2026-11-20_08-50-51/`

The logs show the expected Qwen attempt followed by the cached fallback, but no
complete result file. This exposes a benchmark-adapter/remote-provider stall
that must be repaired before claiming a fresh clean run. It does not invalidate
the completed 40/40 artifact above.

## Next autonomous gates

1. Restore DGX3 reachability from the Pi and run a one-clip ASR/TTS probe.
2. Repair the fdbench adapter stall with a per-trial watchdog and explicit
   assistant-thread error propagation, then rerun all 40 trials.
3. Record clean Qwen ASR/TTS provenance and recompute WER from the new
   `results.json`; do not replace the fallback result until the run is complete.
4. Run the physical human-overlap test on Fifine + USB Audio and record
   barge-in stop latency separately from injected fdbench.
5. Update the deck and ACM paper only with the new provider-specific metrics
   after those gates pass.
