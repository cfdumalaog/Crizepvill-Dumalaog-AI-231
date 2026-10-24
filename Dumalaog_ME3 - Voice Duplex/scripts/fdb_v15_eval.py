"""Evaluate recorded overlap trials against the FULL-DUPLEX-BENCH v1.5 contract.

The live Pi agent and the engineering latency driver are intentionally kept
independent from this evaluator. A recorder can write a JSON object with a
``trials`` list (or pass a bare JSON list), where each trial contains::

    {"scenario": "user_interruption", "behavior": "RESPOND",
     "user_start_s": 7.0, "user_end_s": 8.2,
     "agent_segments": [{"start_s": 5.0, "end_s": 7.4}]}

The evaluator adds the paper timing equations, category accuracy, and
median/p95 stop/response latency. It does not invent missing boundaries and
it does not claim a complete paper run when fewer than the four required
scenario counts are present.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from me3.fdb_v15 import SCENARIOS, score_overlap_trial  # noqa: E402


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "median_s": (float(statistics.median(values)) if values else None),
        "p95_s": percentile(values, 0.95),
    }


def load_trials(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("trials"), list):
        return raw["trials"]
    raise ValueError("input JSON must be a trial list or an object with trials")


def evaluate(trials: list[dict]) -> dict:
    scored: list[dict] = []
    errors: list[dict] = []
    for index, trial in enumerate(trials):
        try:
            scenario = str(trial["scenario"])
            behavior = trial.get("behavior", trial.get("category"))
            result = score_overlap_trial(
                scenario,
                behavior,
                float(trial["user_start_s"]),
                float(trial["user_end_s"]),
                trial.get("agent_segments", []),
            )
            result["trial_id"] = trial.get("trial_id", index)
            for key in ("pre_features", "post_features"):
                if key in trial:
                    result[key] = trial[key]
            scored.append(result)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append({"trial_index": index, "error": str(exc)})

    by_scenario = {}
    for name, spec in SCENARIOS.items():
        rows = [r for r in scored if r["scenario"] == name]
        stops = [r["stop_latency_s"] for r in rows
                 if r["stop_latency_s"] is not None]
        responses = [r["response_latency_s"] for r in rows
                     if r["response_latency_s"] is not None]
        correct = sum(r["behavior"] == spec.expected_behavior for r in rows)
        by_scenario[name] = {
            "expected_behavior": spec.expected_behavior,
            "required_utterances": spec.n_utterances,
            "observed_utterances": len(rows),
            "complete": len(rows) == spec.n_utterances,
            "correct_behavior": correct,
            "behavior_accuracy": (correct / len(rows) if rows else None),
            "categories": {
                category: sum(r["behavior"] == category for r in rows)
                for category in ("RESPOND", "RESUME", "UNCERTAIN", "UNKNOWN")
            },
            "stop_latency": summarize(stops),
            "response_latency": summarize(responses),
            "missing_stop_boundary": sum(
                r["model_stop_s"] is None for r in rows),
            "missing_response_boundary": sum(
                r["model_start_s"] is None for r in rows),
        }

    complete = not errors and all(v["complete"] for v in by_scenario.values())
    return {
        "paper": "FULL-DUPLEX-BENCH v1.5",
        "complete_paper_run": complete,
        "n_input_trials": len(trials),
        "n_scored_trials": len(scored),
        "errors": errors,
        "scenarios": by_scenario,
        "trials": scored,
    }


def markdown(report: dict) -> str:
    lines = [
        "# FULL-DUPLEX-BENCH v1.5 evaluation",
        "",
        f"Complete paper run: **{report['complete_paper_run']}**",
        f"Scored trials: **{report['n_scored_trials']}** / "
        f"{report['n_input_trials']}",
        "",
        "| Scenario | Required | Observed | Expected | Accuracy | "
        "Stop median/p95 (s) | Response median/p95 (s) |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for name, row in report["scenarios"].items():
        stop = row["stop_latency"]
        resp = row["response_latency"]
        acc = "-" if row["behavior_accuracy"] is None else \
            f"{row['behavior_accuracy']:.3f}"
        stop_text = ("-" if stop["median_s"] is None else
                     f"{stop['median_s']:.3f}/{stop['p95_s']:.3f}")
        resp_text = ("-" if resp["median_s"] is None else
                     f"{resp['median_s']:.3f}/{resp['p95_s']:.3f}")
        lines.append(
            f"| {name} | {row['required_utterances']} | "
            f"{row['observed_utterances']} | {row['expected_behavior']} | "
            f"{acc} | {stop_text} | {resp_text} |"
        )
    if report["errors"]:
        lines += ["", "## Invalid trials", ""]
        lines.extend(f"- trial {e['trial_index']}: {e['error']}"
                     for e in report["errors"])
    lines += [
        "",
        "Missing timing boundaries remain missing; this report does not "
        "impute them.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON trial artifact")
    parser.add_argument("--output", type=Path,
                        help="JSON report path (default: beside input)")
    parser.add_argument("--markdown", type=Path,
                        help="Markdown report path (default: beside output)")
    args = parser.parse_args()
    report = evaluate(load_trials(args.input))
    output = args.output or args.input.with_name("fdb_v15_report.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md = args.markdown or output.with_suffix(".md")
    md.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({
        "complete_paper_run": report["complete_paper_run"],
        "n_scored_trials": report["n_scored_trials"],
        "json": str(output),
        "markdown": str(md),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
