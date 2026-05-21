#!/usr/bin/env python
"""Serve a local drag-drop PatchCore gatekeeper app without extra web dependencies."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retinal_ood.inference.gatekeeper import load_patchcore_gatekeeper
from retinal_ood.utils.io import read_yaml

DEFAULT_MAX_UPLOAD_MB = 25


class GatekeeperServer(ThreadingHTTPServer):
    """HTTP server carrying the loaded gatekeeper instance."""

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        gatekeeper: Any,
        max_upload_bytes: int,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.gatekeeper = gatekeeper
        self.max_upload_bytes = max_upload_bytes


class GatekeeperRequestHandler(BaseHTTPRequestHandler):
    server: GatekeeperServer

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", _HTML.encode("utf-8"))
            return
        if self.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "threshold": self.server.gatekeeper.threshold,
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/predict":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("Upload body is empty")
            if content_length > self.server.max_upload_bytes:
                max_mb = self.server.max_upload_bytes / (1024 * 1024)
                raise ValueError(f"Uploaded file is too large; limit is {max_mb:.1f} MB")
            content_type = self.headers.get("Content-Type", "")
            body = self.rfile.read(content_length)
            filename, image_bytes = _parse_image_upload(body, content_type)
            result = self.server.gatekeeper.predict_bytes(filename, image_bytes)
            self._send_json(HTTPStatus.OK, asdict(result))
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive web boundary
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        """Keep uploads quiet so private filenames are not echoed to the terminal."""

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", body)

    def _send_bytes(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _parse_image_upload(body: bytes, content_type: str) -> tuple[str, bytes]:
    if not content_type.startswith("multipart/form-data"):
        raise ValueError("Expected multipart/form-data upload")
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    )
    if not message.is_multipart():
        raise ValueError("Expected multipart upload with an image field")
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        field_name = part.get_param("name", header="content-disposition")
        if field_name != "image":
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            raise ValueError("Uploaded image field is empty")
        filename = Path(part.get_filename() or "upload.png").name
        return filename, payload
    raise ValueError("Upload must include an image file field named 'image'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config used for preprocessing/model setup")
    parser.add_argument("--checkpoint", required=True, help="PatchCore .npz memory-bank checkpoint")
    parser.add_argument("--threshold", type=float, help="Manual operating threshold")
    parser.add_argument(
        "--metrics",
        help="Evaluation metrics.json containing threshold.value; defaults to checkpoint/evaluation/metrics.json",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host; use 0.0.0.0 only behind a tunnel")
    parser.add_argument("--port", type=int, default=7860, help="Bind port")
    parser.add_argument("--max-upload-mb", type=float, default=DEFAULT_MAX_UPLOAD_MB)
    args = parser.parse_args()

    gatekeeper = load_patchcore_gatekeeper(
        read_yaml(args.config),
        args.checkpoint,
        threshold=args.threshold,
        metrics_path=args.metrics,
    )
    max_upload_bytes = int(args.max_upload_mb * 1024 * 1024)
    server = GatekeeperServer(
        (args.host, args.port),
        GatekeeperRequestHandler,
        gatekeeper=gatekeeper,
        max_upload_bytes=max_upload_bytes,
    )
    url = f"http://{args.host}:{args.port}"
    print(f"Serving retinal OOD gatekeeper at {url}")
    if gatekeeper.threshold is None:
        print("Warning: no threshold found; the app will show scores but not ACCEPT/REJECT.")
    server.serve_forever()


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Retinal FAF OOD Gatekeeper</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #18202a;
      --muted: #657286;
      --line: #d8dee8;
      --fill: #f6f8fb;
      --accept: #0f7b50;
      --reject: #b42318;
      --warn: #946200;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }
    main {
      width: min(1120px, calc(100vw - 32px));
      margin: 28px auto;
      display: grid;
      grid-template-columns: minmax(280px, 360px) 1fr;
      gap: 24px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 24px;
      font-weight: 650;
      letter-spacing: 0;
    }
    p { color: var(--muted); margin: 0 0 16px; }
    .drop {
      min-height: 260px;
      border: 2px dashed var(--line);
      background: var(--fill);
      display: grid;
      place-items: center;
      padding: 24px;
      text-align: center;
    }
    .drop.dragover {
      border-color: #3267d6;
      background: #eef4ff;
    }
    button {
      border: 1px solid #1f2937;
      background: #1f2937;
      color: #ffffff;
      padding: 10px 14px;
      cursor: pointer;
      font-weight: 600;
    }
    button:disabled { opacity: 0.6; cursor: wait; }
    input[type="file"] { display: none; }
    .result {
      border-top: 1px solid var(--line);
      padding-top: 16px;
      margin-top: 18px;
    }
    .badge {
      display: inline-block;
      padding: 6px 8px;
      font-weight: 700;
      color: #ffffff;
      background: var(--warn);
    }
    .badge.accept { background: var(--accept); }
    .badge.reject { background: var(--reject); }
    dl {
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 8px 12px;
      margin: 14px 0 0;
    }
    dt { color: var(--muted); }
    dd { margin: 0; overflow-wrap: anywhere; }
    .media {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }
    figure {
      margin: 0;
      border: 1px solid var(--line);
      background: var(--fill);
      min-height: 220px;
    }
    figcaption {
      padding: 9px 10px;
      color: var(--muted);
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }
    img {
      width: 100%;
      height: auto;
      display: block;
    }
    .placeholder {
      min-height: 260px;
      display: grid;
      place-items: center;
      color: var(--muted);
      padding: 18px;
      text-align: center;
    }
    .error { color: var(--reject); margin-top: 12px; }
    @media (max-width: 760px) {
      main { grid-template-columns: 1fr; }
      .media { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Retinal FAF OOD Gatekeeper</h1>
      <p>Binary quality-control decision: valid FAF accept or invalid/OOD reject.</p>
      <div id="drop" class="drop">
        <div>
          <p><strong>Drop an image here</strong></p>
          <button id="choose" type="button">Choose image</button>
          <input id="file" type="file" accept="image/png,image/jpeg,image/tiff,image/bmp" />
        </div>
      </div>
      <div id="error" class="error"></div>
      <div id="result" class="result" hidden>
        <span id="badge" class="badge">WAITING</span>
        <dl>
          <dt>File</dt><dd id="filename"></dd>
          <dt>Score</dt><dd id="score"></dd>
          <dt>Threshold</dt><dd id="threshold"></dd>
          <dt>Decision</dt><dd id="decision"></dd>
        </dl>
      </div>
    </section>
    <section class="media">
      <figure>
        <figcaption>Original</figcaption>
        <div id="originalWrap" class="placeholder">No image loaded</div>
      </figure>
      <figure>
        <figcaption>PatchCore overlay</figcaption>
        <div id="overlayWrap" class="placeholder">No overlay yet</div>
      </figure>
    </section>
  </main>
  <script>
    const drop = document.getElementById("drop");
    const fileInput = document.getElementById("file");
    const choose = document.getElementById("choose");
    const error = document.getElementById("error");
    const result = document.getElementById("result");
    const badge = document.getElementById("badge");
    const originalWrap = document.getElementById("originalWrap");
    const overlayWrap = document.getElementById("overlayWrap");

    choose.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) predict(fileInput.files[0]);
    });
    ["dragenter", "dragover"].forEach((eventName) => {
      drop.addEventListener(eventName, (event) => {
        event.preventDefault();
        drop.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((eventName) => {
      drop.addEventListener(eventName, (event) => {
        event.preventDefault();
        drop.classList.remove("dragover");
      });
    });
    drop.addEventListener("drop", (event) => {
      const file = event.dataTransfer.files[0];
      if (file) predict(file);
    });

    async function predict(file) {
      error.textContent = "";
      result.hidden = true;
      choose.disabled = true;
      originalWrap.innerHTML = "";
      const preview = document.createElement("img");
      preview.src = URL.createObjectURL(file);
      originalWrap.appendChild(preview);
      overlayWrap.className = "placeholder";
      overlayWrap.textContent = "Scoring...";

      const form = new FormData();
      form.append("image", file);
      try {
        const response = await fetch("/predict", { method: "POST", body: form });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Prediction failed");
        renderResult(payload);
      } catch (err) {
        error.textContent = err.message;
        overlayWrap.textContent = "No overlay";
      } finally {
        choose.disabled = false;
      }
    }

    function renderResult(payload) {
      result.hidden = false;
      badge.className = "badge";
      if (payload.decision === "accept_valid_faf") badge.classList.add("accept");
      if (payload.decision === "reject_ood_or_invalid") badge.classList.add("reject");
      badge.textContent = payload.verdict;
      document.getElementById("filename").textContent = payload.filename;
      document.getElementById("score").textContent = payload.score.toFixed(6);
      document.getElementById("threshold").textContent =
        payload.threshold === null ? "not set" : payload.threshold.toFixed(6);
      document.getElementById("decision").textContent = payload.warning || payload.decision;
      overlayWrap.className = "";
      overlayWrap.innerHTML = "";
      const overlay = document.createElement("img");
      overlay.src = "data:image/png;base64," + payload.overlay_png_base64;
      overlayWrap.appendChild(overlay);
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
