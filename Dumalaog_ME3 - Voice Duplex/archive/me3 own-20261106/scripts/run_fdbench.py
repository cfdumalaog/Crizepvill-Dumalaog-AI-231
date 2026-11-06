"""Launch the vendored fd-benchmark with the shared ME3 environment."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
FDBENCH = PROJECT / "benchmarking" / "fd-benchmark"


def main() -> int:
    if not (FDBENCH / "fdbench" / "__main__.py").is_file():
        print(f"fd-benchmark is missing under {FDBENCH}", file=sys.stderr)
        return 2
    env = os.environ.copy()
    # The benchmark uses Rich checkmarks; keep Windows cp1252 consoles from
    # turning a healthy preflight into a UnicodeEncodeError.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(FDBENCH), str(PROJECT), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-m", "fdbench", *sys.argv[1:]],
        cwd=str(FDBENCH), env=env,
    )
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
