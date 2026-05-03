"""Small repository utilities used by the command-line analysis scripts."""

from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np
import torch

def _move_to_repo_root() -> Path:
    """Run relative paths from the repository root."""
    current = Path.cwd().resolve()
    if (current / "data").exists():
        root = current
    elif (current.parent / "data").exists():
        root = current.parent
    else:
        root = current
    os.chdir(root)
    (root / "Figs").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    return root


def display(obj):
    """Small console replacement for IPython.display.display."""
    if hasattr(obj, "to_string"):
        print(obj.to_string())
    else:
        print(obj)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
