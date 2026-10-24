"""FULL-DUPLEX-BENCH v1.5 contract and signal transforms.

This module is deliberately separate from ``benchmark.py``.  The existing
class benchmark measures pipeline latency.  FULL-DUPLEX-BENCH v1.5 measures
what the agent does when user speech overlaps the agent: interruption,
backchannel, talking to others, and background speech.

The constants and equations below mirror the v1.5 paper stored in
``FULL-DUPLEX-BENCH V1.5 - EVALUATING OVERLAP.pdf`` and the checked-in
reference evaluator.  Audio rendering is deterministic and uses the paper's
published gain, filter, reflection, and echo settings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


RESPONSE_CATEGORIES = ("RESPOND", "RESUME", "UNCERTAIN", "UNKNOWN")
EVALUATOR_LABEL_ALIASES = {
    "C_RESPOND": "RESPOND",
    "C_RESUME": "RESUME",
    "C_UNCERTAIN_HANDLING": "UNCERTAIN",
    "C_UNKNOWN": "UNKNOWN",
}


@dataclass(frozen=True)
class ScenarioSpec:
    """One controlled overlap scenario from FULL-DUPLEX-BENCH v1.5."""

    name: str
    expected_behavior: str
    n_utterances: int
    source_voice: str
    description: str


SCENARIOS: dict[str, ScenarioSpec] = {
    "user_interruption": ScenarioSpec(
        "user_interruption", "RESPOND", 200, "primary",
        "A new user request barges in while the agent is speaking.",
    ),
    "user_backchannel": ScenarioSpec(
        "user_backchannel", "RESUME", 99, "primary",
        "A short acknowledgement signals engagement, not a floor takeover.",
    ),
    "talking_to_others": ScenarioSpec(
        "talking_to_others", "RESUME", 100, "primary",
        "The user speaks to another person off-axis while the agent has the floor.",
    ),
    "background_speech": ScenarioSpec(
        "background_speech", "RESUME", 100, "background",
        "A different, distant speaker talks about an unrelated topic.",
    ),
}

# Paper v1.5 signal recipe.
TALKING_OTHER_GAIN_DB = -8.0
TALKING_OTHER_HIGH_SHELF_HZ = 4000.0
TALKING_OTHER_HIGH_SHELF_DB = -5.0
TALKING_OTHER_REFLECTIONS = ((0.045, -6.0), (0.120, -12.0))
BACKGROUND_GAIN_DB = -15.0
BACKGROUND_LOWPASS_HZ = 3000.0
BACKGROUND_ECHO = (0.100, -10.0)

# Paper v1.5 timing definitions.
STOP_LATENCY = "tmodel_stop - tuser_start"
RESPONSE_LATENCY = "tmodel_start - tuser_end"


def _gain(pcm: np.ndarray, db: float) -> np.ndarray:
    return pcm.astype(np.float32, copy=False) * (10.0 ** (db / 20.0))


def _filter(pcm: np.ndarray, sample_rate: int, cutoff_hz: float,
            kind: str) -> np.ndarray:
    """Apply a stable Butterworth filter; scipy is already a ME3 dependency."""
    from scipy.signal import butter, sosfilt

    nyquist = sample_rate / 2.0
    cutoff = min(max(float(cutoff_hz) / nyquist, 1e-4), 0.999)
    sos = butter(4, cutoff, btype=kind, output="sos")
    return sosfilt(sos, pcm.astype(np.float32, copy=False))


def _add_reflections(pcm: np.ndarray, sample_rate: int,
                     reflections: Iterable[tuple[float, float]]) -> np.ndarray:
    """Add delayed copies, retaining the full reflected tail."""
    delays = [(max(0, round(seconds * sample_rate)), db)
              for seconds, db in reflections]
    length = len(pcm) + max((d for d, _ in delays), default=0)
    out = np.zeros(length, dtype=np.float32)
    out[:len(pcm)] += pcm.astype(np.float32, copy=False)
    for delay, db in delays:
        out[delay:delay + len(pcm)] += _gain(pcm, db)
    return out


def _compress(pcm: np.ndarray, threshold: float = 0.35,
              ratio: float = 4.0) -> np.ndarray:
    """Simple deterministic downward compressor for background speech."""
    x = pcm.astype(np.float32, copy=False)
    ax = np.abs(x)
    over = np.maximum(ax - threshold, 0.0)
    compressed = threshold + over / ratio
    return np.sign(x) * np.where(ax > threshold, compressed, ax)


def _clip_int16(pcm: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(pcm), -32768, 32767).astype(np.int16)


def render_talking_to_others(pcm: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Apply the v1.5 off-axis recipe for TALKING_OTHER."""
    x = _gain(pcm, TALKING_OTHER_GAIN_DB)
    low = _filter(x, sample_rate, TALKING_OTHER_HIGH_SHELF_HZ, "lowpass")
    high = _filter(x, sample_rate, TALKING_OTHER_HIGH_SHELF_HZ, "highpass")
    x = low + _gain(high, TALKING_OTHER_HIGH_SHELF_DB)
    return _clip_int16(_add_reflections(x, sample_rate,
                                         TALKING_OTHER_REFLECTIONS))


def render_background_speech(pcm: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Apply the v1.5 distant background-speech recipe."""
    x = _filter(_gain(pcm, BACKGROUND_GAIN_DB), sample_rate,
                BACKGROUND_LOWPASS_HZ, "lowpass")
    x = _add_reflections(x, sample_rate, (BACKGROUND_ECHO,))
    return _clip_int16(_compress(x))


def normalize_response_category(label: str) -> str:
    """Validate a v1.5 categorical evaluator label."""
    value = str(label).strip().upper()
    value = EVALUATOR_LABEL_ALIASES.get(value, value)
    if value not in RESPONSE_CATEGORIES:
        raise ValueError(f"unknown v1.5 response category: {label!r}")
    return value


def stop_latency_s(user_start_s: float, model_stop_s: float | None) -> float | None:
    """Paper equation (1): t_stop = t_model_stop - t_user_start."""
    if model_stop_s is None:
        return None
    return float(model_stop_s) - float(user_start_s)


def response_latency_s(user_end_s: float, model_start_s: float | None) -> float | None:
    """Paper equation (2): t_resp = t_model_start - t_user_end."""
    if model_start_s is None:
        return None
    return float(model_start_s) - float(user_end_s)


def overlap_boundaries_s(
    agent_segments: Sequence[dict], user_start_s: float, user_end_s: float,
) -> tuple[float | None, float | None]:
    """Derive model stop/resume boundaries from VAD speech segments.

    ``agent_segments`` must contain ``start_s`` and ``end_s`` values in one
    monotonic timeline.  A segment crossing the overlap onset supplies the
    stop boundary; the first segment beginning after the overlap supplies the
    resume boundary.  Missing boundaries are represented as ``None`` rather
    than being fabricated.
    """
    segments = sorted(agent_segments, key=lambda s: float(s["start_s"]))
    model_stop = None
    for seg in segments:
        start = float(seg["start_s"])
        end = float(seg["end_s"])
        if start < user_start_s <= end:
            model_stop = end
            break
        if end <= user_start_s:
            model_stop = end
    model_start = next(
        (float(seg["start_s"]) for seg in segments
         if float(seg["start_s"]) >= user_end_s),
        None,
    )
    return model_stop, model_start


def score_overlap_trial(
    scenario: str,
    behavior: str,
    user_start_s: float,
    user_end_s: float,
    agent_segments: Sequence[dict],
) -> dict:
    """Return the paper-v1.5 fields for one recorded overlap trial."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown v1.5 scenario: {scenario!r}")
    behavior = normalize_response_category(behavior)
    stop_s, resume_s = overlap_boundaries_s(
        agent_segments, user_start_s, user_end_s)
    return {
        "scenario": scenario,
        "expected_behavior": SCENARIOS[scenario].expected_behavior,
        "behavior": behavior,
        "user_start_s": float(user_start_s),
        "user_end_s": float(user_end_s),
        "model_stop_s": stop_s,
        "model_start_s": resume_s,
        "stop_latency_s": stop_latency_s(user_start_s, stop_s),
        "response_latency_s": response_latency_s(user_end_s, resume_s),
    }


def contract() -> dict:
    """Machine-readable source contract used by alignment checks and reports."""
    return {
        "paper": "FULL-DUPLEX-BENCH v1.5",
        "scenarios": {
            name: {
                "expected_behavior": spec.expected_behavior,
                "n_utterances": spec.n_utterances,
                "source_voice": spec.source_voice,
            }
            for name, spec in SCENARIOS.items()
        },
        "response_categories": list(RESPONSE_CATEGORIES),
        "reference_evaluator_aliases": dict(EVALUATOR_LABEL_ALIASES),
        "timing_equations": {
            "stop_latency_s": STOP_LATENCY,
            "response_latency_s": RESPONSE_LATENCY,
        },
        "transforms": {
            "talking_to_others": {
                "gain_db": TALKING_OTHER_GAIN_DB,
                "high_shelf_hz": TALKING_OTHER_HIGH_SHELF_HZ,
                "high_shelf_db": TALKING_OTHER_HIGH_SHELF_DB,
                "reflections": list(TALKING_OTHER_REFLECTIONS),
            },
            "background_speech": {
                "gain_db": BACKGROUND_GAIN_DB,
                "lowpass_hz": BACKGROUND_LOWPASS_HZ,
                "echo": BACKGROUND_ECHO,
            },
        },
    }

