"""Compatibility entrypoint: forwards to unified runner (lgbm)."""

from __future__ import annotations

import sys

from runners.train import main


if __name__ == "__main__":
    if "--family" not in sys.argv:
        sys.argv.extend(["--family", "lgbm"])
    main()
