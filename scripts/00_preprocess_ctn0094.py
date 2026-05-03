#!/usr/bin/env python3
"""Command-line entry point for CTN-0094 preprocessing."""

from pathlib import Path
import sys


def _repo_root() -> Path:
    current = Path.cwd().resolve()
    if (current / "data").exists():
        return current
    if (current.parent / "data").exists():
        return current.parent
    return current


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ctn0094_preprocessing import main  # noqa: E402


if __name__ == "__main__":
    main()
