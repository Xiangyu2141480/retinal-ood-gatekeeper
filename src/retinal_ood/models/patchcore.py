"""PatchCore-style detector skeleton.

The full implementation should be developed task-by-task with Codex. This skeleton documents
the required public API and keeps the repository importable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PatchCoreConfig:
    backbone: str = "resnet50"
    layers: tuple[str, ...] = ("layer2", "layer3")
    coreset_ratio: float = 0.1
    nearest_neighbors: int = 1


class PatchCoreDetector:
    """PatchCore detector API placeholder.

    Expected behavior after implementation:
    - fit(dataloader): build a normal patch-feature memory bank from ID data only.
    - predict_scores(dataloader): return anomaly scores and optional heatmaps.
    - save/load: persist memory bank and config.
    """

    def __init__(self, config: PatchCoreConfig) -> None:
        self.config = config
        self.memory_bank: np.ndarray | None = None

    def fit(self, *_: Any, **__: Any) -> None:
        raise NotImplementedError("PatchCoreDetector.fit must be implemented by the Codex task.")

    def predict_scores(self, *_: Any, **__: Any) -> np.ndarray:
        raise NotImplementedError("PatchCoreDetector.predict_scores must be implemented by the Codex task.")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, memory_bank=self.memory_bank, layers=np.array(self.config.layers))

    @classmethod
    def load(cls, path: str | Path) -> "PatchCoreDetector":
        data = np.load(path, allow_pickle=False)
        layers = tuple(str(x) for x in data["layers"].tolist())
        detector = cls(PatchCoreConfig(layers=layers))
        detector.memory_bank = data["memory_bank"]
        return detector
