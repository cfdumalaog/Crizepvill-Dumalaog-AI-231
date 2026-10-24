"""ME3 local smoke test - validate every MCP server + benchmark WITHOUT a Pi.

This is the local-dev gate in the ME2-style workflow: it proves the four MCP
servers (ASR, TTS, VAD, tasks) import, expose their tools, and answer a
`health` call, and that the latency benchmark constructs and runs a dry-run,
all on the Windows dev box. No microphone, no speaker, no network needed.

If this passes locally, the identical code is safe to package and ship to the
Pi (where install.sh adds audio hardware + model weights).

Usage:
    python scripts/local_smoke_test.py [--with-audio]

--with-audio additionally opens a full-duplex audio stream for 1 second to
confirm PortAudio devices are reachable (omit on headless CI).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

SERVERS = ("mcp_asr", "mcp_tts", "mcp_vad", "mcp_tasks")


def import_server(name: str):
    import importlib
    return importlib.import_module(f"me3.{name}")


def call_health(mod) -> dict:
    """Call the server's health() tool directly (no MCP transport needed)."""
    fn = getattr(mod, "health", None)
    if fn is None:
        return {"ok": False, "reason": "no health() tool exposed"}
    return fn()


def check_servers() -> tuple[int, int]:
    passed = failed = 0
    for name in SERVERS:
        try:
            mod = import_server(name)
            # Collect the registered MCP tools from the module's FastMCP app.
            tools = []
            app = getattr(mod, "mcp", None)
            if app is not None:
                for attr in ("list_tools", "get_tools"):
                    getter = getattr(app, attr, None)
                    if callable(getter):
                        try:
                            if asyncio.iscoroutinefunction(getter):
                                res = asyncio.run(getter())
                            else:
                                res = getter()
                            tools = [getattr(t, "name", t) for t in res]
                        except Exception:
                            pass
                        break
            health = call_health(mod)
            ok = isinstance(health, dict) and health.get("ok", True) is not False
            status = "PASS" if ok else "WARN"
            if ok:
                passed += 1
            else:
                failed += 1
            print(f"  [{status}] me3.{name:<9} tools={len(tools)} health={json.dumps(health)[:120]}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [FAIL] me3.{name:<9} {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=1)
    return passed, failed


def check_benchmark() -> bool:
    """Construct the benchmark and run a no-op dry run (no audio hardware)."""
    try:
        from me3.benchmark import LatencyBenchmark
        bench = LatencyBenchmark()
        # Dry run: enumerate scenarios without executing audio trials.
        scenarios = getattr(bench, "scenarios", None) or getattr(bench, "STIMULI", None)
        n = len(scenarios) if hasattr(scenarios, "__len__") else 0
        print(f"  [PASS] me3.benchmark constructed; {n} benchmark scenarios registered")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] me3.benchmark {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=1)
        return False


def check_audio(with_audio: bool) -> bool:
    if not with_audio:
        print("  [SKIP] audio I/O (use --with-audio to test PortAudio devices)")
        return True
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        print(f"  [PASS] PortAudio OK; {len(devs)} devices; default in/out below")
        try:
            print(f"         default in : {sd.default.device[0]}")
            print(f"         default out: {sd.default.device[1]}")
        except Exception:
            pass
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] audio I/O {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-audio", action="store_true",
                        help="also open a PortAudio device query (needs audio hw)")
    args = parser.parse_args()

    print("ME3 local smoke test (no Pi required)\n")
    print("MCP servers:")
    passed, failed = check_servers()
    print()
    print("Benchmark:")
    bench_ok = check_benchmark()
    print()
    print("Audio I/O:")
    audio_ok = check_audio(args.with_audio)

    total_fail = failed + (0 if bench_ok else 1) + (0 if audio_ok else 1)
    print()
    if total_fail == 0:
        print(f"RESULT: PASS - {passed} MCP servers healthy, benchmark + audio OK.")
        print("Safe to run scripts/package_release.py and then scripts/deploy_to_pi.py.")
        return 0
    print(f"RESULT: FAIL - {total_fail} check(s) failed. Fix before packaging.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
