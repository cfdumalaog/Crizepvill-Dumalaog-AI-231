# ME3 + Full-Duplex-Bench integration

ME3 now vendors the public course benchmark at
`benchmarking/fd-benchmark/`. The source is preserved with its MIT `LICENSE`,
and the integration keeps the benchmark's datasets, scoring code, v1/v1.5
scenario definitions, reports, and direct-feed protocol in the project rather
than reimplementing its metrics.

Source provenance: [trueshotjames/ai231-me3-full-duplex-benchmarking](https://github.com/trueshotjames/ai231-me3-full-duplex-benchmarking), commit
`106ac8e543a5fd996e254a87e69fa101bfd27a42` (captured 2026-11-04). The
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
  --brain remote --variant current

# Same tool loop, with Qwen Instruct ASR/TTS:
& $py scripts\me3_fdbench_agent.py --url ws://<laptop-ip>:8791 `
  --brain remote --variant qwen_instruct
```

The benchmark adapter is intentionally separate from the Pi browser speaker
transport. Direct-feed evaluation has one sample clock and no physical
acoustic feedback; the Pi demo has the real Fifine/USB microphone and can route
TTS either to a Pi output device or to the connected browser over LAN.

## Evidence boundary

`--check` and the simulation prove that the vendored benchmark and ME3 adapter
are importable and protocol-compatible. They are not a human-microphone or
paper-grade result. A publishable ME3 result must record the exact variant,
brain/provider, benchmark commit, dataset revision, round count, and generated
`results.json`/`results.csv`/HTML report.
