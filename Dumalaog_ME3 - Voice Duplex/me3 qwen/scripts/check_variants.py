"""Import-check the active ME3 Qwen Instruct package in an isolated interpreter."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

def main() -> int:
    probe = (
        "import me3.agent, me3.audio_io, me3.agent_status, me3.web_audio; "
        "print('variant imports PASS')"
    )
    env = os.environ.copy()
    env["ME3_CODE_VARIANT"] = "qwen"
    env["ME3_VARIANT"] = "qwen_instruct"
    env["PYTHONPATH"] = str(PROJECT)
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=PROJECT, env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        print(f"me3 qwen: FAIL\n{result.stderr[-1000:]}")
        return result.returncode or 1
    print(f"me3 qwen: {result.stdout.strip()}")
    print("ME3 QWEN IMPORTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
