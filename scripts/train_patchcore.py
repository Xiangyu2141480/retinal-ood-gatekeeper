#!/usr/bin/env python
"""Train PatchCore detector.

This script is a scaffold. See docs/CODEX_TASKS.md Task 4.3 for implementation details.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retinal_ood.utils.io import read_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    config = read_yaml(args.config)
    run_name = config.get("project", {}).get("run_name", "patchcore_run")
    runs_dir = Path(config.get("output", {}).get("runs_dir", "runs")) / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)
    write_json(runs_dir / "resolved_config.json", config)
    raise NotImplementedError("PatchCore training is scaffolded; implement via docs/CODEX_TASKS.md Task 4.3")


if __name__ == "__main__":
    main()
