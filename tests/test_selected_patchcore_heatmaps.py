from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _load_script_module():
    script_path = Path("scripts/generate_selected_patchcore_heatmaps.py")
    spec = importlib.util.spec_from_file_location("generate_selected_patchcore_heatmaps", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_select_category_rows_prefers_high_scoring_ood_examples():
    module = _load_script_module()
    scores = pd.DataFrame(
        [
            {"image_path": "id.png", "label": 0, "ood_type": "id", "score": 0.8},
            {
                "image_path": "sensory_low.png",
                "label": 1,
                "ood_type": "sensory_artifact",
                "score": 0.6,
            },
            {
                "image_path": "sensory_high.png",
                "label": 1,
                "ood_type": "sensory_artifact",
                "score": 0.9,
            },
            {
                "image_path": "modality.png",
                "label": 1,
                "ood_type": "modality_shift",
                "score": 0.7,
            },
        ]
    )

    selected = module._select_category_rows(
        scores,
        categories=["sensory_artifact", "modality_shift"],
        per_category=1,
    )

    assert selected["image_path"].tolist() == ["sensory_high.png", "modality.png"]


def test_cli_help_imports_in_fresh_process():
    result = subprocess.run(
        [sys.executable, "scripts/generate_selected_patchcore_heatmaps.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--scores-csv" in result.stdout
