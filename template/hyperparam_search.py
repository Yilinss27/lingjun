"""Deprecated entrypoint.

Hyperparameter search scripts were moved to `template/legacy/`.
Use the new unified pipeline first (`python runners/train.py ...`),
then re-introduce search on top of the new runners.
"""

from __future__ import annotations


def main() -> None:
    print("[Deprecated] hyperparameter search moved to: template/legacy/hyperparam_search_legacy.py")


if __name__ == "__main__":
    main()
