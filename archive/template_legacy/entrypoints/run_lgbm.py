"""Compatibility entrypoint: forwards to unified runner (lgbm)."""

from __future__ import annotations

import sys
import os

_TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TEMPLATE_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from template.runners.train import main


if __name__ == "__main__":
    if "--family" not in sys.argv:
        sys.argv.extend(["--family", "lgbm"])
    main()
