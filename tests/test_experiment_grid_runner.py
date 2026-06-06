from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image


def _load_runner_module():
    script_path = Path("scripts/run_experiment_grid.py")
    spec = importlib.util.spec_from_file_location("run_experiment_grid", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(90, 90, 90)).save(path)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_base_config(path: Path, *, model_name: str = "patchcore", run_name: str = "base") -> Path:
    config = {
        "project": {"name": "toy", "run_name": run_name, "seed": 1},
        "data": {
            "root_dir": "data",
            "train_manifest": "data/manifests/train.csv",
            "val_manifest": "data/manifests/val.csv",
            "test_id_manifest": "data/manifests/test_id.csv",
            "test_ood_manifest": "data/manifests/test_ood.csv",
            "image_size": 8,
        },
        "model": {"name": model_name, "device": "auto", "layers": ["layer2"]},
        "output": {"runs_dir": "runs", "save_heatmaps": False},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _write_grid_config(path: Path, patchcore_config: Path, autoencoder_config: Path | None = None) -> Path:
    experiments: dict[str, dict[str, object]] = {
        "patchcore_layer2": {
            "type": "patchcore",
            "config": patchcore_config.as_posix(),
            "layers": ["layer2"],
        }
    }
    if autoencoder_config is not None:
        experiments["autoencoder_baseline"] = {
            "type": "autoencoder",
            "config": autoencoder_config.as_posix(),
        }
    path.write_text(yaml.safe_dump({"experiments": experiments}), encoding="utf-8")
    return path


def _write_manifests(root: Path) -> tuple[Path, Path, Path, Path]:
    _write_png(root / "data" / "images" / "id" / "train.png")
    _write_png(root / "data" / "images" / "id" / "val.png")
    _write_png(root / "data" / "images" / "id" / "test.png")
    _write_png(root / "data" / "images" / "ood" / "colour.png")
    _write_png(root / "data" / "images" / "ood" / "natural.png")
    train = _write_manifest(
        root / "data" / "manifests" / "train.csv",
        [
            {
                "image_path": "images/id/train.png",
                "label": 0,
                "split": "train",
                "source": "synthetic_faf",
                "ood_type": "id",
                "patient_id": "",
            }
        ],
    )
    val = _write_manifest(
        root / "data" / "manifests" / "val.csv",
        [
            {
                "image_path": "images/id/val.png",
                "label": 0,
                "split": "val",
                "source": "synthetic_faf",
                "ood_type": "id",
                "patient_id": "",
            }
        ],
    )
    test_id = _write_manifest(
        root / "data" / "manifests" / "test_real_id.csv",
        [
            {
                "image_path": "images/id/test.png",
                "label": 0,
                "split": "test",
                "source": "synthetic_faf",
                "ood_type": "id",
                "patient_id": "",
            }
        ],
    )
    test_ood = _write_manifest(
        root / "data" / "manifests" / "test_ood.csv",
        [
            {
                "image_path": "images/ood/colour.png",
                "label": 1,
                "split": "test",
                "source": "public_ood",
                "ood_type": "modality_shift",
                "ood_subtype": "colour_fundus",
                "patient_id": "",
            },
            {
                "image_path": "images/ood/natural.png",
                "label": 1,
                "split": "test",
                "source": "public_ood",
                "ood_type": "semantic_outlier",
                "ood_subtype": "natural_cifar10",
                "patient_id": "",
            },
        ],
    )
    return train, val, test_id, test_ood


def test_dry_run_builds_commands_without_subprocess(tmp_path: Path, monkeypatch):
    module = _load_runner_module()
    _write_manifests(tmp_path)
    patchcore_config = _write_base_config(tmp_path / "configs" / "patchcore.yaml")
    grid_config = _write_grid_config(tmp_path / "grid.yaml", patchcore_config)
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_grid(
        grid_config=grid_config,
        root_dir=tmp_path / "data",
        out_dir=tmp_path / "reports" / "grid",
        dry_run=True,
        only=["patchcore_layer2"],
        device="cpu",
        seed=7,
    )

    assert calls == []
    assert len(result.commands) == 2
    assert result.commands[0][1].endswith("train_patchcore.py")
    assert result.commands[1][1].endswith("evaluate.py")


def test_run_grid_calls_subprocess_and_writes_aggregate_tables(tmp_path: Path, monkeypatch):
    module = _load_runner_module()
    train, val, test_id, test_ood = _write_manifests(tmp_path)
    patchcore_config = _write_base_config(tmp_path / "configs" / "patchcore.yaml")
    grid_config = _write_grid_config(tmp_path / "grid.yaml", patchcore_config)
    calls: list[list[str]] = []

    def fake_run(cmd, check):  # noqa: ANN001
        calls.append(list(cmd))
        config_path = Path(cmd[cmd.index("--config") + 1])
        config = yaml.safe_load(config_path.read_text())
        run_dir = Path(config["output"]["runs_dir"]) / config["project"]["run_name"]
        if cmd[1].endswith("train_patchcore.py"):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "patchcore_memory.npz").write_text("fake", encoding="utf-8")
        if cmd[1].endswith("evaluate.py"):
            eval_dir = run_dir / "evaluation"
            eval_dir.mkdir(parents=True, exist_ok=True)
            (eval_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "global": {
                            "auroc": 0.9,
                            "auprc": 0.8,
                            "fpr_at_95_tpr": 0.1,
                        },
                        "threshold": {"value": 0.5, "source": "validation_id_quantile"},
                        "counts": {"total": 3, "id": 1, "ood": 2},
                        "confusion_matrix": {"tn": 1, "fp": 0, "fn": 0, "tp": 2},
                        "per_ood_type": {
                            "modality_shift": {
                                "auroc": 0.8,
                                "auprc": 0.7,
                                "fpr_at_95_tpr": 0.2,
                                "id_count": 1,
                                "ood_count": 1,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "image_path": "images/id/test.png",
                        "label": 0,
                        "ood_type": "id",
                        "score": 0.1,
                        "prediction": 0,
                        "threshold": 0.5,
                    },
                    {
                        "image_path": "images/ood/colour.png",
                        "label": 1,
                        "ood_type": "modality_shift",
                        "score": 0.9,
                        "prediction": 1,
                        "threshold": 0.5,
                    },
                    {
                        "image_path": "images/ood/natural.png",
                        "label": 1,
                        "ood_type": "semantic_outlier",
                        "score": 0.8,
                        "prediction": 1,
                        "threshold": 0.5,
                    },
                ]
            ).to_csv(eval_dir / "scores.csv", index=False)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    out_dir = tmp_path / "reports" / "grid"
    result = module.run_grid(
        grid_config=grid_config,
        root_dir=tmp_path / "data",
        out_dir=out_dir,
        device="cpu",
        seed=7,
    )

    assert len(calls) == 2
    assert (out_dir / "manifest_audit.json").exists()
    assert (out_dir / "metrics_summary.csv").exists()
    assert (out_dir / "metrics_summary.md").exists()
    assert (out_dir / "per_ood_type_metrics.csv").exists()
    assert (out_dir / "per_ood_subtype_metrics.csv").exists()
    summary = pd.read_csv(out_dir / "metrics_summary.csv")
    assert summary.loc[0, "experiment"] == "patchcore_layer2"
    assert summary.loc[0, "auroc"] == 0.9
    assert summary.loc[0, "id_false_rejection_rate"] == 0.0
    assert summary.loc[0, "id_count"] == 1
    assert summary.loc[0, "ood_count"] == 2
    assert "synthetic fallback" in summary.loc[0, "warnings"]
    subtype = pd.read_csv(out_dir / "per_ood_subtype_metrics.csv")
    assert set(subtype["ood_subtype"]) == {"colour_fundus", "natural_cifar10"}
    assert result.summary_csv == out_dir / "metrics_summary.csv"


def test_skip_existing_avoids_subprocess(tmp_path: Path, monkeypatch):
    module = _load_runner_module()
    _write_manifests(tmp_path)
    patchcore_config = _write_base_config(tmp_path / "configs" / "patchcore.yaml")
    grid_config = _write_grid_config(tmp_path / "grid.yaml", patchcore_config)
    existing = tmp_path / "reports" / "grid" / "runs" / "patchcore_layer2" / "evaluation"
    existing.mkdir(parents=True)
    (existing / "metrics.json").write_text(
        json.dumps(
            {
                "global": {"auroc": 1.0, "auprc": 1.0, "fpr_at_95_tpr": 0.0},
                "threshold": {"value": 0.5},
                "counts": {"id": 1, "ood": 1},
                "confusion_matrix": {"tn": 1, "fp": 0, "fn": 0, "tp": 1},
                "per_ood_type": {},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"image_path": "images/id/test.png", "label": 0, "ood_type": "id", "score": 0.1},
            {
                "image_path": "images/ood/colour.png",
                "label": 1,
                "ood_type": "modality_shift",
                "score": 0.9,
            },
        ]
    ).to_csv(existing / "scores.csv", index=False)

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))

    module.run_grid(
        grid_config=grid_config,
        root_dir=tmp_path / "data",
        out_dir=tmp_path / "reports" / "grid",
        skip_existing=True,
    )

    assert (tmp_path / "reports" / "grid" / "metrics_summary.csv").exists()
