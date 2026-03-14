#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_experiment_comparison import main


if __name__ == "__main__":
    print(
        "[compat] tabpfn/plot_comparison.py now forwards to feature-augmentation plotting via tabpfn/plot_experiment_comparison.py",
        file=sys.stderr,
    )
    main(default_experiment_name="feature_augmentation")
