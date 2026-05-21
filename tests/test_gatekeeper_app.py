import base64
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from retinal_ood.inference.gatekeeper import (
    SingleImageGatekeeper,
    resolve_threshold,
)


class FixedDetector:
    def __init__(self, score: float = 0.75) -> None:
        self.score = score

    def predict_scores(self, dataloader, *, return_patch_maps: bool = False):
        batch = next(iter(dataloader))
        images = batch[0]
        assert torch.is_tensor(images)
        assert images.shape == (1, 3, 8, 8)
        scores = np.array([self.score], dtype=float)
        if return_patch_maps:
            return scores, [np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)]
        return scores


def _png_bytes(color: int = 128) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 10), color=(color, color, color)).save(buffer, format="PNG")
    return buffer.getvalue()


def _load_server_script():
    script_path = Path("scripts/serve_gatekeeper_app.py")
    spec = importlib.util.spec_from_file_location("serve_gatekeeper_app", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_image_gatekeeper_rejects_above_threshold_and_returns_overlay():
    gatekeeper = SingleImageGatekeeper(
        {"data": {"image_size": 8, "grayscale_to_rgb": True, "normalize": "none"}},
        FixedDetector(score=0.75),
        threshold=0.5,
        display_max_side=16,
    )

    result = gatekeeper.predict_bytes("upload.png", _png_bytes())

    assert result.decision == "reject_ood_or_invalid"
    assert result.verdict.startswith("REJECT")
    assert result.prediction == 1
    assert result.score == 0.75
    assert result.threshold == 0.5
    assert base64.b64decode(result.overlay_png_base64 or "").startswith(b"\x89PNG")
    assert base64.b64decode(result.heatmap_png_base64 or "").startswith(b"\x89PNG")


def test_single_image_gatekeeper_warns_when_threshold_missing():
    gatekeeper = SingleImageGatekeeper(
        {"data": {"image_size": 8, "grayscale_to_rgb": True, "normalize": "none"}},
        FixedDetector(score=0.25),
        threshold=None,
        display_max_side=16,
    )

    result = gatekeeper.predict_bytes("upload.png", _png_bytes())

    assert result.decision == "score_only_no_threshold"
    assert result.prediction is None
    assert "No threshold" in (result.warning or "")


def test_single_image_gatekeeper_rejects_unsupported_extension():
    gatekeeper = SingleImageGatekeeper(
        {"data": {"image_size": 8, "grayscale_to_rgb": True, "normalize": "none"}},
        FixedDetector(),
        threshold=0.5,
    )

    with pytest.raises(ValueError, match="Unsupported image extension"):
        gatekeeper.predict_bytes("upload.pdf", _png_bytes())


def test_resolve_threshold_prefers_manual_then_metrics(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"threshold": {"value": 0.42}}), encoding="utf-8")

    assert resolve_threshold({}, threshold=0.9, metrics_path=metrics_path) == 0.9
    assert resolve_threshold({}, metrics_path=metrics_path) == 0.42
    assert resolve_threshold({"evaluation": {"threshold": 0.33}}) == 0.33


def test_parse_image_upload_from_multipart_body():
    server_script = _load_server_script()
    image_bytes = _png_bytes()
    boundary = "----codex-test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="image"; filename="sample.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode("utf-8")
    body += image_bytes
    body += f"\r\n--{boundary}--\r\n".encode("utf-8")

    filename, payload = server_script._parse_image_upload(
        body,
        f"multipart/form-data; boundary={boundary}",
    )

    assert filename == "sample.png"
    assert payload == image_bytes
