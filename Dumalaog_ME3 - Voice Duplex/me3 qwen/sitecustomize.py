"""Add this self-contained variant's importable ``me3`` package to ``sys.path``."""
from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
variant_root = PROJECT / "me3"
if variant_root.is_dir() and str(variant_root) not in sys.path:
    sys.path.insert(0, str(variant_root))
