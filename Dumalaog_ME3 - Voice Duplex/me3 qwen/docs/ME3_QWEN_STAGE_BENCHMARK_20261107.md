# ME3 Qwen Instruct stage-latency benchmark

Date: 2026-11-07 (Asia/Manila)
Variant: ME3 Qwen Instruct on the Raspberry Pi
Benchmark: vendored `ai231-me3-full-duplex-benchmarking`, direct-feed stage block
Raw run: `benchmarking/fd-benchmark/runs/dumalaog-me3-qwen-stage/2026-11-07_07-38-07/`

## Method

The run followed the class protocol: 10 reference phrases, 3 repetitions per
phrase, one stage-mark clock on the Pi, and injected 16-kHz mono PCM. The Pi
opened no physical microphone or speaker for this measurement. The laptop
hosted the benchmark collector and retained the raw JSON, CSV, HTML, per-trial
recordings, `marks.jsonl`, and `stage_latency.json` artifacts.

The benchmark mode disabled tools, external agent calls, fillers, and the live
backchannel. It retained every trial. Latency rows use all valid observations
for that metric and report arithmetic mean and p95. The run contained 30
planned trials, 29 valid reply trials, and one no-reply trial. The no-reply
trial is reported separately and was not turned into a zero or silently
discarded from the run count.

Targets from the updated class benchmark are E2E 4.0 s, ASR final 0.8 s, LLM
TTFT 1.5 s, TTS TTFB 0.6 s, barge-in 0.8 s, and WER 15%.

## Aggregate result

| Metric | Target | Mean | p95 | Valid n | Result |
|---|---:|---:|---:|---:|---|
| E2E latency | < 4.00 s | 5.81 s | 7.87 s | 29 | over target |
| ASR final | < 0.80 s | 3.38 s | 4.92 s | 29 | over target |
| LLM TTFT | < 1.50 s | 0.21 s | 0.21 s | 29 | target met |
| TTS TTFB | < 0.60 s | 1.60 s | 2.98 s | 28 | over target |
| Barge-in | < 0.80 s | not measured | not measured | 0 | separate overlap run required |
| WER | < 15% | 29.3% | not applicable | 29 | over target |

Because each per-run group has fewer than 20 observations, the benchmark flags
per-run p95 as close to the slowest trial. The headline aggregate uses all 30
planned trials and the valid observations available for each metric. No best
run was selected.

## Provider evidence boundary

The Pi VPN was active during the run: `openvpn-client@sharchpc.service` was
active, `tun0` was present, and the route to DGX3 used the VPN interface. DGX3
TTS port 8001 responded to the protected endpoint, but DGX3 ASR port 8000 did
not accept TCP connections from either the Pi or the VPN-connected laptop.
The ME3 runtime therefore used its cached multilingual faster-whisper base
fallback for ASR. The results above are valid diagnostic stage-latency evidence
for the deployed pipeline, but they are not a Qwen2-Audio ASR performance
claim. Repeat the same 10 × 3 run after port 8000 is restored to obtain the
clean Qwen Instruct audio result.

The raw benchmark output is intentionally kept outside the release package by
the benchmark's ignore policy. The checked-in method and summary are the
reproducible record; the raw run directory contains the machine-readable
evidence used to produce this table.
