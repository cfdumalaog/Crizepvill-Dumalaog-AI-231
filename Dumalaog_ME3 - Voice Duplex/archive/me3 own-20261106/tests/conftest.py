from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from me3_bootstrap import configure_me3_path  # noqa: E402

configure_me3_path(PROJECT)
