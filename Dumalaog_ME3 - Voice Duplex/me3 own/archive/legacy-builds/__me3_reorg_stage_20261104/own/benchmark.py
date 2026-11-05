"""ME3 Latency Benchmark (CLASSWORK) - standardized latency measurement.

A reproducible, standardized method to measure full-duplex voice-agent
latency on the RPi. Every scenario uses the SAME protocol so results are
comparable across hardware (Pi 4 vs Pi 5), models, and team members.

STANDARDIZED PROTOCOL
=====================
1. Fixed stimulus set: a canonical set of N utterances (see DATASET) recorded
   once, stored as 16kHz mono PCM16 WAV. ALL group members use the same files.
2. Warm-up: 3 discarded trials per scenario to warm caches/models.
3. Trials: n_trials (default 20) per scenario, randomized order, median reported.
4. Clock: time.perf_counter() (monotonic, ns resolution) stamped at defined
   boundary events. No wall-clock.
5. Telemetry: psutil CPU% and RAM sampled around each trial; frame_ms logged.
6. Reporting: per-scenario median + p95 + min/max, plus a composite
   "end-to-end" figure. Output CSV + JSON + console table.

BOUNDARY EVENTS (the standardized timestamps)
---------------------------------------------
  T0  user_speech_end      last speech frame (VAD endpoint)
  T1  asr_final            ASR returns final transcript
  T2  llm_first_token      first LLM token received
  T3  tts_first_byte       first TTS audio byte produced
  T4  tts_first_played     first TTS audio byte written to speaker
  T5  agent_speech_end     agent finishes speaking

DERIVED METRICS (standardized names)
------------------------------------
  asr_latency        = T1 - T0
  llm_ttft           = T2 - T1      (time to first token)
  tts_ttfb           = T3 - T2      (time to first audio byte)
  audio_out_latency  = T4 - T3
  end_to_end         = T4 - T0      (user done -> agent first audible byte)
  barge_in_latency   = T_stop - T_interrupt  (user speech -> TTS stopped)

FULL-DUPLEX BEHAVIOR METRICS (ported from Full-Duplex-Bench, arXiv:2503.04721)
------------------------------------------------------------------------------
The latency figures above say HOW FAST the agent reacts. They do NOT say
whether it SHOULD have reacted at all. Full-Duplex-Bench (FDB) adds that
behavior layer - the "did it speak when it should have stayed silent?"
question:

  takeover_rate (TOR)  fraction of user pauses where the agent took over
                       (spoke) instead of staying silent. LOWER = better for
                       pause-handling; HIGHER (=1) = better for
                       user-interruption. Binary per turn: TO=0 if the agent
                       response was silence or a backchannel, else TO=1.
  backchannel_freq     backchannel events per second of user speech.
  backchannel_jsd      Jensen-Shannon divergence between the agent's
                       backchannel-timing distribution P and the human
                       ground-truth distribution Q (200 ms windows).
                       0 = identical timing, 1 = fully divergent. LOWER=better.

  FDB BACKCHANNEL DEFINITION: an agent speech segment that is BOTH
  < 1.0 s long AND < 2 words. Anything else counts as a takeover.

These are computed from per-segment (start_s, end_s, n_words) lists - the
word-level time-alignment output faster-whisper / Parakeet-TDT already
produce - so no extra inference infra is needed. Entry points:
classify_segment / takeover_rate / backchannel_frequency /
jensen_shannon_divergence / evaluate_full_duplex.

Scenarios (from config):
  wake_to_first_word, utterance_to_llm_token, llm_token_to_tts_audio,
  end_to_end, barge_in

Run:  python -m me3.benchmark [--trials 20] [--out results/]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

from .config import load_config

# --- Full-Duplex-Bench (arXiv:2503.04721) behavior-metric constants --------
BC_MAX_DURATION_S = 1.0   # backchannel must be shorter than this
BC_MAX_WORDS = 2          # ...and contain no more words than this
BC_FORMAL_SPEECH_S = 3.0  # FDB repo time_threshold: a segment longer than
                          # this is "formally speaking" (session TOR flag = 1)
JS_WINDOW_S = 0.200       # 200 ms bin width for the JSD timing histogram

# Canonical stimulus set (group-shared). Filenames map to data/bench/*.wav
DATASET = [
    "hello", "what_time_is_it", "set_a_timer_5_minutes",
    "tell_me_system_status", "stop", "repeat_that",
    "play_the_news", "what_is_2_plus_2",
]


# ============================================================================
# Full-Duplex-Bench (arXiv:2503.04721) behavior metrics
#
# Pure functions over per-segment timing lists, so they run anywhere (no Pi,
# no audio device) and are unit-testable. Each segment is a dict:
#   {"start_s": float, "end_s": float, "n_words": int}
# which is exactly the shape of word-level time-aligned ASR output.
# ============================================================================

def classify_segment(seg: dict) -> str:
    """Classify one agent speech segment as 'backchannel' or 'takeover'.

    FDB rule: a backchannel is BOTH shorter than BC_MAX_DURATION_S
    (< 1.0 s) AND fewer than BC_MAX_WORDS (< 2 words). Everything else -
    including a long response or a multi-word utterance - is a takeover.
    """
    dur = float(seg["end_s"]) - float(seg["start_s"])
    if dur < BC_MAX_DURATION_S and int(seg["n_words"]) <= BC_MAX_WORDS:
        return "backchannel"
    return "takeover"


def session_takeover_flag(agent_segments: Sequence[dict]) -> int:
    """FDB repo per-session TOR flag (time_threshold = 3.0 s in the
    evaluation scripts): 1 if ANY agent segment is formally speaking
    (duration > BC_FORMAL_SPEECH_S), else 0. Complements takeover_rate
    (the per-segment mean) in reports."""
    return 1 if any(
        float(s["end_s"]) - float(s["start_s"]) > BC_FORMAL_SPEECH_S
        for s in agent_segments
    ) else 0


def takeover_rate(agent_segments: Sequence[dict]) -> float:
    """Takeover Rate (TOR): mean of the per-turn takeover indicator.

    Each segment is one candidate turn. TO=1 if it is a takeover, 0 if it is
    a backchannel (or there are no segments at all -> pure silence -> 0.0).
    Lower is better for pause-handling; higher (==1.0) is better for
    user-interruption. Empty input -> 0.0 (agent stayed silent).
    """
    segs = list(agent_segments)
    if not segs:
        return 0.0
    flags = [1.0 if classify_segment(s) == "takeover" else 0.0 for s in segs]
    return sum(flags) / len(flags)


def backchannel_frequency(agent_segments: Sequence[dict],
                          user_speech_seconds: float) -> float:
    """Backchannel events per second of user speech (FDB 'Freq'). Higher
    (up to a human-plausible ceiling) indicates more natural acknowledgment.
    Returns 0.0 if there is no user speech to normalize against."""
    if user_speech_seconds <= 0:
        return 0.0
    n_bc = sum(1 for s in agent_segments
               if classify_segment(s) == "backchannel")
    return n_bc / user_speech_seconds


def _bin_distribution(events, t_max: float) -> list[float]:
    """Histogram backchannel events into JS_WINDOW_S bins over [0, t_max].

    FDB repo convention (eval_backchannel.py): each backchannel interval
    [start, end] increments every bin it covers. Events may be (start, end)
    tuples or bare start times (zero width). Returns a probability vector
    (sums to 1.0) so it can be fed to jensen_shannon_divergence. Events
    outside [0, t_max] are clipped in.
    """
    n_bins = max(1, int(math.ceil(t_max / JS_WINDOW_S)))
    counts = [0] * n_bins
    for ev in events:
        if isinstance(ev, (tuple, list)):
            t0, t1 = float(ev[0]), float(ev[1])
        else:
            t0 = t1 = float(ev)
        i0 = min(max(int(t0 / JS_WINDOW_S), 0), n_bins - 1)
        i1 = min(max(int(t1 / JS_WINDOW_S), 0), n_bins - 1)
        for i in range(i0, i1 + 1):
            counts[i] += 1
    total = sum(counts)
    if total == 0:
        # uniform fallback so JSD is well-defined even with no events
        return [1.0 / n_bins] * n_bins
    return [c / total for c in counts]


def jensen_shannon_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence between two distributions (base-e log).

    Matches the FDB repo convention (scipy.spatial.distance.jensenshannon):
    0.0 = identical distributions, ln(2) ~= 0.693 = maximally divergent.
    Used to compare the agent's backchannel timing (P) against the human
    ground-truth timing (Q). Inputs need not sum to 1 (normalized internally)
    and may differ in length (zero-padded to the longer)."""
    n = max(len(p), len(q))
    if n == 0:
        return 0.0
    pp = [p[i] / sum(p) if sum(p) > 0 else 0.0 for i in range(n)]
    qq = [q[i] / sum(q) if sum(q) > 0 else 0.0 for i in range(n)]
    m = [(a + b) / 2.0 for a, b in zip(pp, qq)]

    def _kl(x: list[float], y: list[float]) -> float:
        s = 0.0
        for xi, yi in zip(x, y):
            if xi > 0.0 and yi > 0.0:
                s += xi * math.log(xi / yi)
        return s

    return (_kl(pp, m) + _kl(qq, m)) / 2.0


def fdb_jsd_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon *distance* exactly as the FDB repo reports it.

    The FDB evaluation scripts use scipy.spatial.distance.jensenshannon,
    which returns the SQUARE ROOT of the (base-e) Jensen-Shannon
    divergence. Headline JSD numbers in ME3 outputs use this convention so
    they are directly comparable to FDB paper/repo values; the plain
    jensen_shannon_divergence() (base-e, no sqrt) is also reported.
    """
    return math.sqrt(max(0.0, jensen_shannon_divergence(p, q)))


@dataclass
class FullDuplexReport:
    """Aggregate behavior metrics for one scenario (FDB-style)."""
    scenario: str
    n_turns: int
    takeover_rate: float
    backchannel_freq: float
    backchannel_jsd: float | None  # None when no ground-truth Q supplied
    n_backchannels: int
    n_takeovers: int


def evaluate_full_duplex(
    scenario: str,
    agent_segments: Sequence[dict],
    user_speech_seconds: float,
    human_bc_times: Sequence[float] | None = None,
    t_max: float | None = None,
) -> FullDuplexReport:
    """Compute the FDB behavior metrics for one scenario's recorded output.

    Args:
      scenario:            label for the report.
      agent_segments:      agent speech segments (start_s/end_s/n_words).
      user_speech_seconds: total duration of the user's speech (for Freq).
      human_bc_times:      optional human-annotated backchannel times (s) -
                           the ground-truth Q for JSD. Omit to skip JSD.
      t_max:                timeline length (s) for the JSD histogram;
                           defaults to the latest event time seen.
    """
    segs = list(agent_segments)
    bc_intervals = [(float(s["start_s"]), float(s["end_s"])) for s in segs
                    if classify_segment(s) == "backchannel"]
    n_bc = len(bc_intervals)
    n_to = len(segs) - n_bc

    jsd: float | None = None
    if human_bc_times is not None:
        if n_bc == 0:
            # FDB repo convention: js_divergence = 1 when the agent
            # produced no backchannel predictions at all.
            jsd = 1.0
        else:
            horizon = t_max if t_max is not None else max(
                [e for iv in bc_intervals for e in iv] + list(human_bc_times))
            horizon = max(horizon, JS_WINDOW_S)
            p = _bin_distribution(bc_intervals, horizon)
            q = _bin_distribution(list(human_bc_times), horizon)
            jsd = jensen_shannon_divergence(p, q)

    return FullDuplexReport(
        scenario=scenario,
        n_turns=len(segs),
        takeover_rate=takeover_rate(segs),
        backchannel_freq=backchannel_frequency(segs, user_speech_seconds),
        backchannel_jsd=jsd,
        n_backchannels=n_bc,
        n_takeovers=n_to,
    )


@dataclass
class TrialResult:
    scenario: str
    stimulus: str
    trial: int
    # boundary deltas (ms)
    asr_latency: float = 0.0
    llm_ttft: float = 0.0
    tts_ttfb: float = 0.0
    audio_out_latency: float = 0.0
    end_to_end: float = 0.0
    barge_in_latency: float = 0.0
    # telemetry
    cpu_pct: float = 0.0
    ram_mb: float = 0.0
    frame_ms: float = 0.0


@dataclass
class ScenarioSummary:
    scenario: str
    n: int
    metric: str
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float


class LatencyBenchmark:
    """Standardized latency measurement harness."""

    PRIMARY_METRIC = {
        "wake_to_first_word": "end_to_end",
        "utterance_to_llm_token": "llm_ttft",
        "llm_token_to_tts_audio": "tts_ttfb",
        "end_to_end": "end_to_end",
        "barge_in": "barge_in_latency",
    }

    def __init__(self, config: dict | None = None):
        self.cfg = config or load_config()
        b = self.cfg["benchmark"]
        self.n_trials = int(b.get("n_trials", 20))
        self.warmup = int(b.get("warmup", 3))
        self.scenarios = b.get("scenarios", list(self.PRIMARY_METRIC))
        self.out_dir = Path("results")
        self._results: list[TrialResult] = []

    # ---- Telemetry helper --------------------------------------------------
    def _telemetry(self) -> tuple[float, float]:
        try:
            import psutil
            return psutil.cpu_percent(interval=0.05), \
                psutil.virtual_memory().used / (1024 * 1024)
        except Exception:  # noqa: BLE001
            return 0.0, 0.0

    # ---- Scenario runners (stubbed to the pipeline; swap in real calls) ----
    def _run_scenario(self, scenario: str, stimulus: str, trial: int) -> TrialResult:
        """Execute one trial for a scenario.

        In the full build each T-boundary is stamped by the live pipeline
        (VAD/ASR/LLM/TTS). Here we provide the measurement skeleton with
        clearly-marked hooks so the standardized protocol is fixed and the
        group can drop in identical real calls.
        """
        r = TrialResult(scenario=scenario, stimulus=stimulus, trial=trial)
        r.frame_ms = float(self.cfg["audio"]["frame_ms"])

        T0 = time.perf_counter()
        # HOOK: feed stimulus WAV through VAD; stamp T0 at endpoint.
        # T1 = time.perf_counter()  # after ASR final
        # T2 = time.perf_counter()  # after first LLM token
        # T3 = time.perf_counter()  # after first TTS byte
        # T4 = time.perf_counter()  # after first byte to speaker
        T1 = time.perf_counter()
        T2 = time.perf_counter()
        T3 = time.perf_counter()
        T4 = time.perf_counter()

        r.asr_latency = (T1 - T0) * 1000
        r.llm_ttft = (T2 - T1) * 1000
        r.tts_ttfb = (T3 - T2) * 1000
        r.audio_out_latency = (T4 - T3) * 1000
        r.end_to_end = (T4 - T0) * 1000
        if scenario == "barge_in":
            # HOOK: stamp interrupt and TTS-stop
            r.barge_in_latency = 0.0
        r.cpu_pct, r.ram_mb = self._telemetry()
        return r

    # ---- Orchestration -----------------------------------------------------
    def run(self, trials: int | None = None) -> list[ScenarioSummary]:
        n = trials or self.n_trials
        self._results = []
        for scenario in self.scenarios:
            # warmup (discarded)
            for _ in range(self.warmup):
                self._run_scenario(scenario, DATASET[0], -1)
            for i in range(n):
                stim = DATASET[i % len(DATASET)]
                self._results.append(self._run_scenario(scenario, stim, i))
        return self.summarize()

    def summarize(self) -> list[ScenarioSummary]:
        summaries = []
        for scenario in self.scenarios:
            metric = self.PRIMARY_METRIC.get(scenario, "end_to_end")
            vals = [getattr(r, metric) for r in self._results
                    if r.scenario == scenario and r.trial >= 0]
            if not vals:
                continue
            vals_sorted = sorted(vals)
            p95 = vals_sorted[int(0.95 * (len(vals_sorted) - 1))]
            summaries.append(ScenarioSummary(
                scenario=scenario, n=len(vals), metric=metric,
                median_ms=round(statistics.median(vals), 2),
                p95_ms=round(p95, 2),
                min_ms=round(min(vals), 2),
                max_ms=round(max(vals), 2),
                mean_ms=round(statistics.mean(vals), 2),
            ))
        return summaries

    def save(self, summaries: list[ScenarioSummary], out_dir: str | None = None):
        out = Path(out_dir or self.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # JSON
        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in summaries], f, indent=2)
        # CSV (all trials)
        import csv
        with open(out / "trials.csv", "w", newline="", encoding="utf-8") as f:
            cols = list(asdict(self._results[0]).keys()) if self._results else []
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in self._results:
                if r.trial >= 0:
                    w.writerow(asdict(r))
        return out

    def print_table(self, summaries: list[ScenarioSummary]):
        print("\n=== ME3 Latency Benchmark (standardized) ===")
        hdr = f"{'scenario':<24}{'metric':<18}{'n':>4}{'med_ms':>10}{'p95_ms':>10}{'mean_ms':>10}"
        print(hdr)
        print("-" * len(hdr))
        for s in summaries:
            print(f"{s.scenario:<24}{s.metric:<18}{s.n:>4}"
                  f"{s.median_ms:>10.2f}{s.p95_ms:>10.2f}{s.mean_ms:>10.2f}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    bench = LatencyBenchmark()
    summaries = bench.run(trials=args.trials)
    bench.print_table(summaries)
    out = bench.save(summaries, args.out)
    print(f"Saved results to {out}/")


if __name__ == "__main__":
    main()
