"""Static source-contract and Raspberry Pi readiness check for ME3.

This check intentionally does not claim that a live Pi run happened.  It
verifies that the checked-in implementation exposes the contracts required by
the papers placed in the ME3 folder and that the release contains the Linux
installer, systemd units, ARM-safe model overrides, and all four MCP servers.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from me3.config import load_config  # noqa: E402
from me3.fdb_v15 import (  # noqa: E402
    RESPONSE_CATEGORIES,
    SCENARIOS,
    contract,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    cfg = load_config()
    source_pdfs = [
        PROJECT / "Building AI Voice Agents - Summary.pdf",
        PROJECT / "FULL-DUPLEX-BENCH V1.5 - EVALUATING OVERLAP.pdf",
        PROJECT / "refs" / "FDB_paper_2503.04721.pdf",
    ]
    for pdf in source_pdfs:
        require(pdf.is_file(), f"source paper missing: {pdf}")

    fdb = cfg["fdb_v15"]
    require(list(fdb["scenarios"]) == list(SCENARIOS),
            "config fdb_v15.scenarios drifted from the paper order")
    require(fdb["response_categories"] == list(RESPONSE_CATEGORIES),
            "response categories must be RESPOND/RESUME/UNCERTAIN/UNKNOWN")
    for name, spec in SCENARIOS.items():
        require(int(fdb["stimulus_counts"][name]) == spec.n_utterances,
                f"{name}: v1.5 stimulus count mismatch")
    transforms = fdb["signal_transforms"]
    require(transforms["talking_to_others"]["gain_db"] == -8,
            "talking-to-others gain drifted")
    require(transforms["talking_to_others"]["high_shelf_hz"] == 4000,
            "talking-to-others high-shelf cutoff drifted")
    require(transforms["talking_to_others"]["high_shelf_db"] == -5,
            "talking-to-others high-shelf attenuation drifted")
    require(transforms["background_speech"]["gain_db"] == -15,
            "background-speech gain drifted")
    require(transforms["background_speech"]["lowpass_hz"] == 3000,
            "background-speech low-pass cutoff drifted")

    # The Pi path is intentionally checked as a release contract, not as a
    # claim that hardware was online during this run.
    require((PROJECT / "install.sh").is_file(), "Pi installer missing")
    require((PROJECT / "requirements.txt").is_file(), "Pi requirements missing")
    require(all((PROJECT / "deploy" / f"me3-{name}.service").is_file()
                for name in ("asr", "tts", "vad", "agent")),
            "one or more Pi systemd units are missing")
    require((PROJECT / "scripts" / "fdb_v15_eval.py").is_file(),
            "paper-native v1.5 evaluator missing")
    require(all((PROJECT / "me3" / f"{name}.py").is_file()
                for name in ("mcp_asr", "mcp_tts", "mcp_vad", "mcp_tasks")),
            "one or more owned MCP servers are missing")
    require(cfg["audio"]["full_duplex"] is True,
            "full_duplex must remain enabled")
    require(cfg["asr"]["compute_type"] == "int8",
            "Pi deployment must use int8 ASR")
    require(os.path.basename(str(PROJECT)) == "Dumalaog_ME3 - Voice Duplex",
            "project folder must retain the ME3 name")

    print("PAPER ALIGNMENT PASS")
    print(json.dumps({
        "papers": [str(p.relative_to(PROJECT)) for p in source_pdfs],
        "fdb_v15": contract(),
        "pi": {
            "installer": True,
            "systemd_units": 4,
            "owned_mcp_servers": 4,
            "full_duplex": True,
            "asr_compute_type": cfg["asr"]["compute_type"],
        },
    }, indent=2))
    print("LIVE_PI_RUN: not performed by this static check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

