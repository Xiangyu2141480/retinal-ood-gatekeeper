#!/usr/bin/env python
"""Evaluate a trained OOD detector.

This script is a scaffold. See docs/CODEX_TASKS.md Task 5.1 for implementation details.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    _ = parser.parse_args()
    raise NotImplementedError("Evaluation CLI is scaffolded; implement via docs/CODEX_TASKS.md Task 5.1")


if __name__ == "__main__":
    main()
