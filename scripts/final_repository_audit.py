#!/usr/bin/env python
"""Final repository audit for dissertation delivery.

The audit checks packaging and safety invariants. It does not validate model quality or run
training; those claims are tied to the committed result tables and reproducibility runbooks.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".slurm",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

REQUIRED_FINAL_DOCS = [
    Path("docs/dissertation/final_evidence_index.md"),
    Path("docs/dissertation/final_figure_shortlist.md"),
    Path("docs/dissertation/final_table_shortlist.md"),
    Path("docs/dissertation/reproducibility_runbook.md"),
    Path("docs/dissertation/school_server_runbook.md"),
    Path("docs/dissertation/claims_and_limitations_matrix.md"),
    Path("docs/dissertation/README.md"),
]

REQUIRED_FINAL_REPORTS = [
    Path("reports/dissertation_final/final_result_summary.md"),
    Path("reports/dissertation_final/final_result_summary.csv"),
    Path("reports/dissertation_final/final_figure_index.md"),
    Path("reports/dissertation_final/final_table_index.md"),
    Path("reports/dissertation_final/final_reproducibility_checklist.md"),
    Path("reports/dissertation_final/final_claim_evidence_index.md"),
    Path("reports/dissertation_final/final_appendix_index.md"),
]

REQUIRED_SOURCE_EVIDENCE = [
    Path("reports/dissertation_figures/figure_index.md"),
    Path("reports/dissertation_figures/robustness/figure_index.md"),
    Path("reports/dissertation_results/multi_scheme_comparison/metrics_by_scheme.csv"),
    Path("reports/dissertation_results/robustness_analysis/result_provenance.md"),
    Path("reports/dissertation_results/robustness_analysis/dissertation_takeaway_table.md"),
    Path("datasets/dissertation_v1/manifests/train_id.csv"),
    Path("datasets/dissertation_v1/manifests/test_ood_full.csv"),
    Path("datasets/dissertation_v1/manifests/test_id_synthetic_fallback.csv"),
]

WINDOWS_ABSOLUTE_PATH_RE = r"(?<![a-z])[a-z]:(?:\\+|/(?!/))"
PRIVATE_PATH_TOKENS = ["/" + "Users/", "Documents" + "/Codex"]
PRIVATE_OVERLEAF_RE = "git\\.overleaf\\.com/" + "[a-z0-9]+"
PRIVATE_PATH_RE = re.compile(
    "(?i)(?:"
    + WINDOWS_ABSOLUTE_PATH_RE
    + "|"
    + "|".join(re.escape(token) for token in PRIVATE_PATH_TOKENS)
    + "|"
    + PRIVATE_OVERLEAF_RE
    + ")"
)
SECRET_RE = re.compile(
    r"(?i)((api[_-]?key|secret|password|token)\s*[:=]\s*\S+|"
    r"gh[opsu]_[A-Za-z0-9_]+|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|"
    r"BEGIN (RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY)"
)
CHECKPOINT_RE = re.compile(r"\.(pt|pth|ckpt|npz)$", re.IGNORECASE)


class AuditResult:
    def __init__(self, failures: list[str], warnings: list[str]) -> None:
        self.failures = failures
        self.warnings = warnings

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, object]:
        return {"passed": self.passed, "failures": self.failures, "warnings": self.warnings}


def _git_ls_files(repo_root: Path) -> list[str] | None:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _tracked_or_existing_files(repo_root: Path) -> list[Path]:
    tracked = _git_ls_files(repo_root)
    if tracked is not None:
        return [repo_root / path for path in tracked]
    return [path for path in repo_root.rglob("*") if path.is_file()]


def _relative(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _check_required_paths(repo_root: Path, failures: list[str]) -> None:
    for path in REQUIRED_FINAL_DOCS + REQUIRED_FINAL_REPORTS + REQUIRED_SOURCE_EVIDENCE:
        if not (repo_root / path).exists():
            failures.append(f"missing required path: {path.as_posix()}")


def _check_private_paths_and_secrets(repo_root: Path, failures: list[str]) -> None:
    for path in _tracked_or_existing_files(repo_root):
        if not path.is_file() or not _is_text_file(path):
            continue
        rel = _relative(repo_root, path)
        if rel.startswith("reports/generated/") or rel.startswith(".git/"):
            continue
        text = _read_text(path)
        if PRIVATE_PATH_RE.search(text):
            failures.append(f"private path pattern found in {rel}")
        if SECRET_RE.search(text):
            failures.append(f"secret/API-key-like pattern found in {rel}")


def _check_patient_identifiers(repo_root: Path, failures: list[str]) -> None:
    for path in _tracked_or_existing_files(repo_root):
        if path.suffix.lower() != ".csv":
            continue
        rel = _relative(repo_root, path)
        if rel.startswith("reports/generated/"):
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            failures.append(f"could not read CSV for patient identifier audit: {rel}: {exc}")
            continue
        for banned in ["Eye_ID", "clinical_label", "biomarker_label", "disease_label"]:
            if banned in frame.columns:
                failures.append(f"banned clinical/private column {banned} found in {rel}")
        if "patient_id" in frame.columns:
            non_empty = frame["patient_id"].fillna("").astype(str).str.strip().ne("")
            if bool(non_empty.any()):
                failures.append(f"non-empty patient_id values found in {rel}")


def _check_forbidden_tracked_artifacts(repo_root: Path, failures: list[str]) -> None:
    tracked = _git_ls_files(repo_root)
    candidates = tracked if tracked is not None else [_relative(repo_root, path) for path in repo_root.rglob("*")]
    for rel in candidates:
        normalized = rel.replace("\\", "/")
        if CHECKPOINT_RE.search(normalized):
            failures.append(f"tracked checkpoint/model artifact found: {normalized}")
        if normalized == "runs" or normalized.startswith("runs/"):
            failures.append(f"tracked raw run directory found: {normalized}")


def _check_dataset_lfs(repo_root: Path, failures: list[str], warnings: list[str]) -> None:
    attributes = repo_root / ".gitattributes"
    if not attributes.exists() or "data/images/dissertation_v1/**/*.png filter=lfs" not in _read_text(attributes):
        failures.append("Git LFS pattern for dissertation dataset images is missing from .gitattributes")

    tracked = _git_ls_files(repo_root)
    if tracked is not None:
        image_paths = [
            path
            for path in tracked
            if path.replace("\\", "/").startswith("data/images/dissertation_v1/")
            and Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        ]
    else:
        image_paths = [
            _relative(repo_root, path)
            for path in (repo_root / "data/images/dissertation_v1").rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        ]
    if not image_paths:
        failures.append("no dissertation dataset image files found/tracked under data/images/dissertation_v1")

    try:
        completed = subprocess.run(
            ["git", "lfs", "ls-files"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        warnings.append("git lfs ls-files could not be run; checked .gitattributes and image presence only")
        return
    if completed.returncode == 0 and "data/images/dissertation_v1" not in completed.stdout:
        warnings.append("git lfs ls-files did not list dissertation images; verify LFS pointers on server clone")


def _check_readme_warnings(repo_root: Path, failures: list[str]) -> None:
    readme_paths = [repo_root / "README.md", repo_root / "docs/dissertation/README.md"]
    combined = "\n".join(_read_text(path).lower() for path in readme_paths if path.exists())
    if "not a clinical device" not in combined and "not clinical deployment" not in combined:
        failures.append("README warning missing: not a clinical device / not clinical deployment")
    if "not a disease classifier" not in combined:
        failures.append("README warning missing: not a disease classifier")


def _check_final_index_references(repo_root: Path, failures: list[str]) -> None:
    index_paths = [
        repo_root / "reports/dissertation_final/final_figure_index.md",
        repo_root / "reports/dissertation_final/final_table_index.md",
        repo_root / "reports/dissertation_final/final_appendix_index.md",
    ]
    for index_path in index_paths:
        if not index_path.exists():
            continue
        for line in _read_text(index_path).splitlines():
            if not line.startswith("|"):
                continue
            cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
            if not cells:
                continue
            candidate = cells[0]
            if not candidate.startswith(("docs/", "reports/", "scripts/", "datasets/", "data/")):
                continue
            if "*" in candidate:
                continue
            if not (repo_root / candidate).exists():
                failures.append(
                    f"referenced path missing in {_relative(repo_root, index_path)}: {candidate}"
                )


def run_audit(repo_root: str | Path = ".", write_json: str | Path | None = None) -> AuditResult:
    repo_root = Path(repo_root)
    failures: list[str] = []
    warnings: list[str] = []

    _check_required_paths(repo_root, failures)
    _check_private_paths_and_secrets(repo_root, failures)
    _check_patient_identifiers(repo_root, failures)
    _check_forbidden_tracked_artifacts(repo_root, failures)
    _check_dataset_lfs(repo_root, failures, warnings)
    _check_readme_warnings(repo_root, failures)
    _check_final_index_references(repo_root, failures)

    result = AuditResult(failures=failures, warnings=warnings)
    if write_json is not None:
        path = Path(write_json)
        if not path.is_absolute():
            path = repo_root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the final dissertation repository audit.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--write-json", default=None)
    args = parser.parse_args()

    result = run_audit(repo_root=args.repo_root, write_json=args.write_json)
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    if result.failures:
        for failure in result.failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("Final repository audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
