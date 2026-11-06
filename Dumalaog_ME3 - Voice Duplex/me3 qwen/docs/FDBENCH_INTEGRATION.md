# ME3 + Full-Duplex-Bench integration

ME3 now vendors the public course benchmark at
`benchmarking/fd-benchmark/`. The source is preserved with its MIT `LICENSE`,
and the integration keeps the benchmark's datasets, scoring code, v1/v1.5
scenario definitions, reports, and direct-feed protocol in the project rather
than reimplementing its metrics.

Source provenance: [trueshotjames/ai231-me3-full-duplex-benchmarking](https://github.com/trueshotjames/ai231-me3-full-duplex-benchmarking), commit
`57eaddbc188ce9a19b8fd002b575f703426956f4` (captured 2026-11-06). The
upstream README describes direct feed as 16-kHz mono int16 PCM over WebSocket,
with a JSON `{"event":"clear"}` message for interrupted output; the ME3
adapter below implements that protocol.

## Safe smoke checks

```powershell
$py = '..\..\.venv\Scripts\python.exe'
& $py scripts\run_fdbench.py --help
& $py scripts\run_fdbench.py check
& $py scripts\me3_fdbench_agent.py --check
```

The integrated simulation does not open a microphone or speaker:

```powershell
& $py scripts\run_fdbench.py run --sim --datasets datasets-dev `
  --student me3-smoke --rounds 1 --no-asr
```

The result remains a benchmark artifact under
`benchmarking/fd-benchmark/runs/`; generated runs are ignored from release
packages. For the actual ME3 agent lane, start the benchmark on the laptop,
then connect the selected ME3 variant with:

```powershell
& $py scripts\me3_fdbench_agent.py --url ws://<laptop-ip>:8791 `
  --brain remote --variant qwen_instruct
```

The benchmark adapter is intentionally separate from the Pi browser speaker
transport. Direct-feed evaluation has one sample clock and no physical
acoustic feedback; the Pi demo has the real Fifine/USB microphone and can route
TTS either to a Pi output device or to the connected browser over LAN.

For the class stage-latency rows, the adapter exposes the project's optional
stage marks. ME3 marks VAD end, ASR final, streamed LLM first token, TTS request,
first TTS audio, audio output, and interruption. The marks use the Pi monotonic
clock and travel to the laptop benchmark over the benchmark UDP mark channel.
Benchmark mode sets the acknowledgment wait to zero, so a slow turn does not
add the live assistant's backchannel to the latency measurement.

## Evidence boundary

`--check` and the simulation prove that the vendored benchmark and ME3 adapter
are importable and protocol-compatible. They are not a human-microphone or
paper-grade result. A publishable ME3 result must record the exact variant,
brain/provider, benchmark commit, dataset revision, round count, and generated
`results.json`/`results.csv`/HTML report.

## Class latency protocol (updated 2026-11-06)

The current upstream benchmark uses direct feed by default: the assistant
receives injected 16-kHz mono PCM over the benchmark socket and opens no real
microphone or speaker. The updated upstream run adds the stage-latency block,
stage marks, latency chart, and the agreed targets. ME3's adapter additionally
defaults to benchmark mode, which omits tool schemas, external agent/harness
dispatch, filler/backchannel audio, and real channel side effects. Use
`--allow-tools` only for a separate agentic demonstration.

Targets: E2E 4.0 s, ASR final 0.8 s, LLM TTFT 1.5 s, TTS TTFB 0.6 s, and
barge-in 0.8 s. Report the arithmetic mean and p95 across every valid trial in
the predefined run set. Do not select the fastest run. With fewer than 20
trials, label p95 as close to the slowest trial. If the ASR provider is
unavailable, keep the run as diagnostic evidence and label the fallback in the
deck rather than presenting it as the Qwen ASR result.
