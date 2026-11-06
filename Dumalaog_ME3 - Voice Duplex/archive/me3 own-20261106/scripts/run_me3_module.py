"""Run an importable ME3 module after selecting ``me3 own`` or ``me3 qwen``.

The project contains a self-contained ``me3`` package, but this wrapper keeps
the documented Pi/Windows commands explicit and reliable from any directory.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from me3_bootstrap import configure_me3_path  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        print("usage: python scripts/run_me3_module.py me3.agent [module args ...]")
        return 2
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    configure_me3_path(PROJECT)
    runpy.run_module(module, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
