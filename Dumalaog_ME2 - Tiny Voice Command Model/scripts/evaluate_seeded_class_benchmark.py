"""Run the class holdout WAV bundle locally on the Raspberry Pi models.

This bypasses laptop-to-Pi live speaker streaming and microphone capture. It
tests the exported ONNX pair and the production confidence/margin gates on the
same wake-plus-command WAVs prepared by the class benchmark. It does not run
GPIO/actions and does not measure acoustic microphone performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import threading
import time
import wave
from pathlib import Path
from urllib.request import urlopen

import numpy as np


def load_runtime(project_root: Path, benchmark_root: Path):
    sys.path.insert(0, str(benchmark_root.resolve()))
    sys.path.insert(0, str(project_root))
    from vcm_app import (  # noqa: PLC0415
        BENCHMARK_INTENT_SLOT,
        INTENT_CONFIDENCE_THRESHOLD,
        INTENT_MARGIN_THRESHOLD,
        VCMPredictor,
        accepts_intent_prediction,
    )
    try:
        from tinyvcm_model.config import SAMPLES, SR  # noqa: PLC0415
        from tinyvcm_model.frontend import fit_audio  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        # The live Pi release packages this same frontend as tinyvcm_runtime.
        # Keep the local training/export project compatible while evaluating
        # the exact installed runtime without copying code into the deployment.
        if exc.name != "tinyvcm_model":
            raise
        from tinyvcm_runtime.config import SAMPLES, SR  # noqa: PLC0415
        from tinyvcm_runtime.frontend import fit_audio  # noqa: PLC0415
    return (BENCHMARK_INTENT_SLOT, INTENT_CONFIDENCE_THRESHOLD, INTENT_MARGIN_THRESHOLD,
            VCMPredictor, accepts_intent_prediction, SAMPLES, SR, fit_audio)


def read_pcm16_mono(path: Path, sample_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        rate = stream.getframerate()
        width = stream.getsampwidth()
        frames = stream.getnframes()
        raw = stream.readframes(frames)
    if channels != 1 or rate != sample_rate or width != 2:
        raise ValueError(f"Expected mono PCM16 at {sample_rate} Hz: {path}")
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def bounded_segment(audio: np.ndarray, start_s: float, end_s: float, sr: int) -> np.ndarray:
    start = max(0, min(len(audio), round(start_s * sr)))
    end = max(start, min(len(audio), round(end_s * sr)))
    return audio[start:end]


def live_endpoint_window(audio: np.ndarray, endpoint_s: float, sr: int, samples: int) -> np.ndarray:
    """Recreate the rolling live buffer 350 ms after the spoken endpoint."""
    end = min(len(audio), max(0, round((endpoint_s + 0.35) * sr)))
    start = max(0, end - samples)
    window = audio[start:end]
    if len(window) < samples:
        window = np.pad(window, (samples - len(window), 0))
    return window.astype(np.float32, copy=False)


def wake_max_probability(predictor, audio: np.ndarray, samples: int, fit_audio) -> dict:
    """Score five temporal placements, matching the benchmark wake evaluator."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size > samples:
        audio = fit_audio(audio, target_samples=samples)
    max_start = max(0, samples - len(audio))
    starts = np.linspace(0, max_start, 5, dtype=int)
    preds = []
    for start in starts:
        window = np.zeros(samples, dtype=np.float32)
        window[start:start + len(audio)] = audio
        preds.append(predictor.predict(window, mode="wake"))
    best = max(preds, key=lambda prediction: prediction["wake_probability"])
    return {
        "label": "WAKE_WORD" if best["wake_probability"] >= predictor.wake_threshold else "NON_WAKE",
        "wake_probability": float(best["wake_probability"]),
        "latency_ms": float(statistics.mean(p["latency_ms"] for p in preds)),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def device_state() -> dict:
    result = {"hostname": platform.node(), "platform": platform.platform(), "python": platform.python_version()}
    try:
        with urlopen("http://127.0.0.1:7860/api/state", timeout=2) as response:
            state = json.loads(response.read().decode("utf-8"))
        result["microphone"] = {
            "name": state.get("microphone"),
            "requested_device": state.get("microphone_requested_device"),
            "stream_active": state.get("microphone_stream_active"),
            "audio_age_sec": state.get("audio_age_sec"),
        }
        result["assistant_model_source"] = state.get("model_source_run")
    except Exception as exc:  # live UI may be intentionally stopped
        result["assistant_state_error"] = f"{type(exc).__name__}: {exc}"
    return result


class PiSampler:
    """Sample Pi resources while local WAV inference runs; no audio is played."""
    def __init__(self, benchmark_root: Path, interval_sec: float = 0.5):
        sys.path.insert(0, str(benchmark_root.resolve()))
        import pi_agent  # noqa: PLC0415

        self.agent = pi_agent
        self.interval_sec = interval_sec
        self.samples: list[dict] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.previous_cpu = None
        self.previous_proc: dict[str, tuple[int, float]] = {}
        self.clock_hz = os.sysconf("SC_CLK_TCK")
        self.own_pid = os.getpid()
        self.assistant_pids = self._find_assistant_pids()
        self.specs = self.agent.build_specs()

    @staticmethod
    def _find_assistant_pids() -> list[int]:
        found = []
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            pid = int(name)
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            except OSError:
                continue
            if "vcm_app.py" in cmd and "--port 7860" in cmd and pid != os.getpid():
                found.append(pid)
        return found

    def _sample(self) -> dict:
        now = time.perf_counter()
        cpu = self.agent.read_cpu_stat()
        system_pct = None
        if cpu and self.previous_cpu and len(cpu) == len(self.previous_cpu):
            system_pct = self.agent.pct(self.previous_cpu[0], cpu[0])
        self.previous_cpu = cpu

        mi = self.agent.meminfo()
        total_kb, avail_kb = mi.get("MemTotal"), mi.get("MemAvailable")
        sample = {
            "t": now,
            "temp_c": self.agent.read_temp(),
            "cpu_pct": system_pct,
            "freq_mhz": self.agent.read_freq_mhz(),
            "load1": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
            "mem_used_mb": self.agent.kb_to_mb(total_kb - avail_kb) if total_kb and avail_kb is not None else None,
            "mem_avail_mb": self.agent.kb_to_mb(avail_kb),
            "throttled": self.agent.get_throttled(),
        }
        for key, pid in (("proc", self.own_pid), ("assistant_proc", self.assistant_pids[0] if self.assistant_pids else None)):
            info = self.agent.proc_stat(pid) if pid else None
            if info:
                ticks, rss_mb, threads = info
                old = self.previous_proc.get(key)
                cpu_pct = None
                if old and now > old[1]:
                    cpu_pct = round((ticks - old[0]) / self.clock_hz / (now - old[1]) * 100.0, 1)
                self.previous_proc[key] = (ticks, now)
                sample[key] = {
                    "cpu_pct": cpu_pct,
                    "rss_mb": rss_mb,
                    "cpu_time_s": ticks / self.clock_hz,
                    "threads": threads,
                }
            else:
                sample[key] = None
        self.samples.append(sample)
        return sample

    def start(self) -> float:
        self._sample()
        self.thread = threading.Thread(target=self._run, name="pi-benchmark-sampler", daemon=True)
        self.thread.start()
        return time.perf_counter()

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_sec):
            self._sample()

    def stop(self) -> float:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=self.interval_sec + 1)
        self._sample()
        return time.perf_counter()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True, help="Folder with plan.json, audio/, and wake/")
    parser.add_argument("--project-root", type=Path, required=True, help="Deployed dandan folder with vcm_app.py and tinyvcm_model/")
    parser.add_argument("--benchmark-root", type=Path, required=True, help="Class benchmark checkout containing vcmbench/ and pi_agent.py")
    parser.add_argument("--intent-model", type=Path, required=True)
    parser.add_argument("--wake-model", type=Path, required=True)
    parser.add_argument("--wake-threshold", type=float, default=0.95)
    parser.add_argument("--size", choices=("quick", "full"), default="quick")
    parser.add_argument("--limit", type=int, help="Optional debug-only trial limit; omit for the complete selected plan")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    (BENCHMARK_INTENT_SLOT, intent_confidence_threshold, intent_margin_threshold,
     VCMPredictor, accepts_intent_prediction, SAMPLES, SR, fit_audio) = load_runtime(
         args.project_root.resolve(), args.benchmark_root.resolve())
    plan_path = args.bundle_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if args.limit:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        plan = plan[:args.limit]
    elif args.size == "quick" and len(plan) != 125:
        raise ValueError(f"Quick mode requires the prepared 125-trial plan; found {len(plan)} trials")
    elif args.size == "full" and len(plan) != 218:
        raise ValueError(f"Full mode requires the prepared 218-trial plan; found {len(plan)} trials")
    predictor = VCMPredictor(args.intent_model, args.wake_model, args.wake_threshold)
    model_hashes_before = {"intent": sha256(args.intent_model), "wake": sha256(args.wake_model)}
    command_times: list[float] = []
    wake_times: list[float] = []
    trials: list[dict] = []
    sampler = PiSampler(args.benchmark_root)
    specs = sampler.specs
    t_start = sampler.start()

    try:
        for item in plan:
            row = dict(item)
            row.update({
                "pred_intent": "NO_RESPONSE", "pred_slot": "", "pred_variation": "",
                "pred_variation_id": None, "pred_raw": "", "n_command_events": 0,
                "wake_logged": False, "latency_s": None, "infer_ms": None, "audio_ms": None,
            })
            path = args.bundle_dir / item["audio_file"]
            audio = read_pcm16_mono(path, SR)
            kind = item.get("kind", "wake")
            if kind == "wake":
                wake_audio = bounded_segment(audio, item["wake_start"], item["wake_end"], SR)
            else:
                wake_audio = bounded_segment(audio, item["cmd_start"], item["cmd_end"], SR)
            wake_result = wake_max_probability(predictor, wake_audio, SAMPLES, fit_audio)
            wake_times.append(wake_result["latency_ms"])
            row["seeded_wake_probability"] = wake_result["wake_probability"]
            row["seeded_wake_detected"] = wake_result["label"] == "WAKE_WORD"
            row["wake_logged"] = row["seeded_wake_detected"]

            if row["seeded_wake_detected"]:
                window = live_endpoint_window(audio, item["cmd_end"], SR, SAMPLES)
                prediction = predictor.predict(window, mode="command")
                command_times.append(float(prediction["latency_ms"]))
                row["seeded_top_label"] = prediction["label"]
                row["seeded_confidence"] = float(prediction["confidence"])
                row["seeded_margin"] = float(prediction["margin"])
                if accepts_intent_prediction(prediction["confidence"], prediction["margin"]):
                    intent, slot = BENCHMARK_INTENT_SLOT[prediction["label"]]
                    row["pred_intent"] = intent
                    row["pred_slot"] = slot or ""
                    row["pred_raw"] = prediction["label"]
                    row["n_command_events"] = 1
                    row["infer_ms"] = float(prediction["latency_ms"])
                    row["audio_ms"] = SAMPLES * 1000 / SR
            trials.append(row)
    finally:
        t_end = sampler.stop()
    model_hashes_after = {"intent": sha256(args.intent_model), "wake": sha256(args.wake_model)}
    if model_hashes_after != model_hashes_before:
        raise RuntimeError("Model files changed during the evaluation; refusing to publish results")

    from vcmbench.report import score, write_outputs  # noqa: PLC0415

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "seeded_trials.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trials), encoding="utf-8"
    )
    meta = {
        "student": "Dumalaog ME2", "started": time.strftime("%Y%m%d-%H%M%S"),
        "mode": "seeded_local_onnx", "wake_word": "Hi Dandan", "wake_gap": 0.8,
        "size": args.size, "seed": 231, "pi_mic": None, "mic_check": None,
        "pi_log_file": None, "holdout": "airimonda/vcm-benchmark local WAV bundle",
        "execution_note": "Pi-local file inference; no laptop speaker streaming, no microphone capture, no GPIO/actions.",
        "trial_limit": args.limit,
        "plan_sha256": sha256(plan_path),
    }
    metrics = score(trials, sampler.samples, specs, None, t_start, t_end, meta)
    metrics["execution_note"] = meta["execution_note"]
    metrics["models"] = {
        "intent": {"path": str(args.intent_model), "bytes": args.intent_model.stat().st_size, "sha256": sha256(args.intent_model)},
        "wake": {"path": str(args.wake_model), "bytes": args.wake_model.stat().st_size, "sha256": sha256(args.wake_model)},
        "wake_threshold": predictor.wake_threshold,
    }
    metrics["resource_measurement_note"] = (
        "Pi resources sampled at 0.5 s intervals during seeded inference. The process row is the evaluator, "
        "and assistant process fields are preserved separately in seeded_summary.json. This is not an acoustic live run."
    )
    write_outputs(args.output_dir, metrics, trials, [])
    report_path = args.output_dir / "report.md"
    report_path.write_text(
        "# Seeded local Pi ONNX evaluation\n\n"
        "> File-based inference on the Pi using the class benchmark's prepared WAVs. "
        "This bypasses acoustic microphone capture and live speaker playback. It does not measure "
        "end-to-end acoustic response latency or physical VAD behavior. CPU/RAM/temperature samples "
        "describe the Pi during this local evaluator process, not the full live playback SOP.\n\n" + report_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    wake_files = sorted((args.bundle_dir / "wake").glob("*.wav"))
    personal_wake = []
    for path in wake_files:
        result = wake_max_probability(predictor, read_pcm16_mono(path, SR), SAMPLES, fit_audio)
        personal_wake.append({"file": path.name, **result})

    summary = {
        "title": "ME2 - VCM on Raspberry Pi 5 — seeded local holdout evaluation",
        "device": device_state(), "thresholds": {
            "wake": predictor.wake_threshold,
            "intent_confidence": intent_confidence_threshold,
            "intent_margin": intent_margin_threshold,
        },
        "size": args.size, "model_hashes": model_hashes_before,
        "plan_trials": len(plan), "wake_positive_trials": sum(t.get("kind", "wake") == "wake" for t in plan),
        "no_wake_trials": sum(t.get("kind") == "no_wake" for t in plan),
        "wake_performance_ms": {
            "n": len(wake_times), "mean_5_windows": statistics.mean(wake_times) if wake_times else None,
            "p95_5_windows": float(np.percentile(wake_times, 95)) if wake_times else None,
        },
        "command_performance_ms": {
            "n": len(command_times), "mean": statistics.mean(command_times) if command_times else None,
            "p95": float(np.percentile(command_times, 95)) if command_times else None,
        },
        "independent_recorded_wake_takes": personal_wake,
        "pi_resource_samples": len(sampler.samples),
        "pi_evaluator_process": [s.get("proc") for s in sampler.samples if s.get("proc")],
        "pi_assistant_process": [s.get("assistant_proc") for s in sampler.samples if s.get("assistant_proc")],
        "classification": {
            "intent_accuracy": metrics["intent_level"]["accuracy"],
            "intent_accuracy_ci95": metrics["intent_level"]["accuracy_ci95"],
            "command_accuracy": metrics["command_level"]["accuracy"],
            "command_accuracy_ci95": metrics["command_level"]["accuracy_ci95"],
            "false_accepts": metrics["intent_level"]["false_accepts"],
            "out_of_scope_count": metrics["intent_level"]["n_out_of_scope"],
            "false_wakes": metrics["false_wake"]["false_wakes"],
            "no_wake_count": metrics["false_wake"]["n"],
            "per_intent": metrics["intent_level"]["per_class"],
            "confusions": metrics["intent_level"]["confusions"],
        },
    }
    summary["device"].update({
        "intent_model": metrics["models"]["intent"],
        "wake_model": metrics["models"]["wake"],
    })
    (args.output_dir / "seeded_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "host": summary["device"]["hostname"], "size": summary["size"], "plan_trials": summary["plan_trials"],
        "intent_accuracy": summary["classification"]["intent_accuracy"],
        "command_accuracy": summary["classification"]["command_accuracy"],
        "false_accepts": summary["classification"]["false_accepts"],
        "false_wakes": summary["classification"]["false_wakes"],
        "wake_takes": personal_wake,
        "outputs": [str(report_path), str(args.output_dir / "seeded_summary.json")],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
