"""Import-check both self-contained ME3 variant folders in isolated interpreters."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUTER = PROJECT.parent


def main() -> int:
    probe = (
        "import me3.agent, me3.audio_io, me3.agent_status, me3.web_audio; "
        "print('variant imports PASS')"
    )
    for variant_dir in ("me3 own", "me3 qwen"):
        variant_root = OUTER / variant_dir
        env = os.environ.copy()
        env["ME3_CODE_VARIANT"] = "qwen" if variant_dir == "me3 qwen" else "own"
        env["PYTHONPATH"] = str(variant_root)
        result = subprocess.run(
            [sys.executable, "-c", probe], cwd=variant_root, env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            print(f"{variant_dir}: FAIL\n{result.stderr[-1000:]}")
            return result.returncode or 1
        print(f"{variant_dir}: {result.stdout.strip()}")
    print("ME3 VARIANT IMPORTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
