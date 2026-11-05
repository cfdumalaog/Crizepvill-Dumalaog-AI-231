"""Select the code package in this self-contained ME3 variant."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root(start: str | Path | None = None) -> Path:
    here = Path(start or __file__).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if (candidate / "config" / "config.yaml").is_file():
            return candidate
    raise RuntimeError("could not locate the ME3 project root")


def configure_me3_path(root: str | Path | None = None,
                       variant: str | None = None) -> Path:
    """Put this variant's importable ``me3`` package first on ``sys.path``."""
    root_path = project_root(root)
    selected = root_path / "me3"
    if not selected.is_dir():
        raise FileNotFoundError(f"ME3 package directory is missing: {selected}")
    while str(selected) in sys.path:
        sys.path.remove(str(selected))
    sys.path.insert(0, str(selected))
    os.environ["ME3_CODE_VARIANT"] = (
        "qwen" if root_path.name.casefold() == "me3 qwen" else "own"
    )
    if root_path.name.casefold() == "me3 qwen":
        os.environ.setdefault("ME3_VARIANT", "qwen_instruct")
    return selected


if __name__ == "__main__":
    print(configure_me3_path())
