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

from retinal_ood.inference.gatekeeper import SUPPORTED_IMAGE_SUFFIXES, load_patchcore_gatekeeper
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
                _build_health_payload(self.server.gatekeeper, self.server.max_upload_bytes),
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


def _build_health_payload(gatekeeper: Any, max_upload_bytes: int) -> dict[str, Any]:
    config = getattr(gatekeeper, "config", {})
    project_cfg = config.get("project", {}) if isinstance(config, dict) else {}
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    detector = getattr(gatekeeper, "detector", None)
    detector_cfg = getattr(detector, "config", None)
    return {
        "status": "ok",
        "threshold": getattr(gatekeeper, "threshold", None),
        "threshold_set": getattr(gatekeeper, "threshold", None) is not None,
        "supported_extensions": sorted(SUPPORTED_IMAGE_SUFFIXES),
        "max_upload_mb": round(max_upload_bytes / (1024 * 1024), 2),
        "run": {
            "name": project_cfg.get("run_name"),
            "project": project_cfg.get("name"),
        },
        "model": {
            "name": model_cfg.get("name"),
            "backbone": model_cfg.get("backbone", getattr(detector_cfg, "backbone", None)),
            "layers": model_cfg.get("layers", list(getattr(detector_cfg, "layers", []))),
            "image_size": data_cfg.get("image_size"),
        },
        "privacy": {
            "uploads_saved": False,
            "filenames_logged": False,
            "disease_classifier": False,
        },
    }


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
      --ink: #19202a;
      --muted: #667085;
      --line: #d0d7e2;
      --panel: #ffffff;
      --soft: #f4f7fb;
      --soft-2: #edf2f7;
      --accent: #2357c6;
      --accent-soft: #e8f0ff;
      --accept: #0f7b50;
      --accept-soft: #e5f6ee;
      --reject: #b42318;
      --reject-soft: #fde8e5;
      --warn: #946200;
      --warn-soft: #fff3d6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 48%);
    }
    button, input, table { font: inherit; }
    button {
      border: 1px solid #243142;
      background: #243142;
      color: #ffffff;
      padding: 10px 14px;
      cursor: pointer;
      font-weight: 700;
    }
    button.secondary {
      background: #ffffff;
      color: #243142;
      border-color: var(--line);
    }
    button:disabled { opacity: 0.55; cursor: wait; }
    main {
      width: min(1240px, calc(100vw - 32px));
      margin: 28px auto;
      display: grid;
      grid-template-columns: minmax(310px, 390px) 1fr;
      gap: 22px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.12;
      font-weight: 750;
      letter-spacing: 0;
    }
    h2 {
      margin: 0 0 10px;
      font-size: 17px;
      font-weight: 750;
      letter-spacing: 0;
    }
    p { color: var(--muted); margin: 0; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      padding: 18px;
    }
    .stack { display: grid; gap: 16px; }
    .eyebrow {
      color: var(--accent);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .notice {
      background: var(--warn-soft);
      border: 1px solid #f5d48b;
      color: #6b4500;
      padding: 10px 12px;
      font-size: 13px;
    }
    .flow {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .step {
      background: var(--soft);
      border: 1px solid var(--line);
      padding: 10px;
      min-height: 78px;
    }
    .step strong {
      display: block;
      font-size: 13px;
      margin-bottom: 4px;
    }
    .step span { color: var(--muted); font-size: 12px; }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .chip {
      border: 1px solid var(--line);
      background: #ffffff;
      padding: 5px 8px;
      color: #415064;
      font-size: 12px;
      font-weight: 700;
    }
    .drop {
      min-height: 230px;
      border: 2px dashed #b8c5d6;
      background: var(--soft);
      display: grid;
      place-items: center;
      padding: 22px;
      text-align: center;
      transition: border-color 120ms ease, background 120ms ease;
    }
    .drop.dragover {
      border-color: var(--accent);
      background: var(--accent-soft);
    }
    .drop-title {
      margin: 0 0 8px;
      font-size: 19px;
      color: var(--ink);
      font-weight: 750;
    }
    .drop-actions {
      display: flex;
      gap: 8px;
      justify-content: center;
      flex-wrap: wrap;
      margin-top: 16px;
    }
    input[type="file"] { display: none; }
    .status {
      min-height: 24px;
      color: var(--muted);
      font-size: 13px;
    }
    .status.error { color: var(--reject); font-weight: 700; }
    .status.warn { color: var(--warn); font-weight: 700; }
    .summary {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      background: var(--soft);
      border: 1px solid var(--line);
      padding: 12px;
    }
    .metric dt {
      color: var(--muted);
      font-size: 12px;
      margin: 0 0 5px;
    }
    .metric dd {
      margin: 0;
      font-size: 20px;
      font-weight: 780;
    }
    .toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }
    .table-wrap {
      border: 1px solid var(--line);
      overflow: auto;
      max-height: 320px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
      background: #ffffff;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 10px 11px;
      text-align: left;
      vertical-align: middle;
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      background: #f8fafc;
      color: #415064;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      z-index: 1;
    }
    tr.result-row { cursor: pointer; }
    tr.result-row:hover, tr.result-row.selected { background: #f4f8ff; }
    .badge {
      display: inline-block;
      min-width: 78px;
      padding: 5px 8px;
      font-size: 12px;
      font-weight: 800;
      text-align: center;
    }
    .badge.accept { color: var(--accept); background: var(--accept-soft); }
    .badge.reject { color: var(--reject); background: var(--reject-soft); }
    .badge.warn { color: var(--warn); background: var(--warn-soft); }
    .badge.error { color: var(--reject); background: var(--reject-soft); }
    .media {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }
    figure {
      margin: 0;
      border: 1px solid var(--line);
      background: var(--soft);
      min-height: 260px;
    }
    figcaption {
      padding: 10px 12px;
      color: #415064;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
      font-weight: 700;
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
    .details {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 8px 12px;
      margin-top: 12px;
      color: #415064;
      font-size: 13px;
    }
    .details dt { color: var(--muted); }
    .details dd { margin: 0; overflow-wrap: anywhere; }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .media { grid-template-columns: 1fr; }
      .summary { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <section class="stack">
      <div class="panel stack">
        <div>
          <div class="eyebrow">FAF quality-control gatekeeper</div>
          <h1>Retinal FAF OOD Gatekeeper</h1>
          <p>Drop retinal images to decide whether they look like valid FAF input for a downstream model.</p>
        </div>
        <div class="notice">
          This is not a disease classifier. It only returns ACCEPT valid FAF or REJECT OOD / invalid input.
        </div>
        <div class="flow" aria-label="4-step workflow">
          <div class="step"><strong>1. Prepare checkpoint</strong><span>Train PatchCore and run evaluation once.</span></div>
          <div class="step"><strong>2. Drop images</strong><span>Upload one or multiple ordinary image files.</span></div>
          <div class="step"><strong>3. Score</strong><span>Compare anomaly score against the threshold.</span></div>
          <div class="step"><strong>4. Review decision</strong><span>Inspect ACCEPT/REJECT and heatmap overlay.</span></div>
        </div>
        <div class="chips" aria-label="Supported file types">
          <span class="chip">.png</span><span class="chip">.jpg</span><span class="chip">.jpeg</span>
          <span class="chip">.tif</span><span class="chip">.tiff</span><span class="chip">.bmp</span>
        </div>
      </div>
      <div class="panel stack">
        <div id="drop" class="drop">
          <div>
            <p class="drop-title">Drop image files here</p>
            <p>Multiple files are scored one by one. Images are processed in memory only.</p>
            <div class="drop-actions">
              <button id="choose" type="button">Choose images</button>
              <button id="clear" class="secondary" type="button">Clear</button>
            </div>
            <input id="file" type="file" accept="image/png,image/jpeg,image/tiff,image/bmp" multiple />
          </div>
        </div>
        <div id="status" class="status">Ready. Supported files: PNG, JPEG, TIFF, BMP.</div>
      </div>
    </section>
    <section class="stack">
      <div class="summary" aria-label="Result counts">
        <dl class="metric"><dt>Total</dt><dd id="totalCount">0</dd></dl>
        <dl class="metric"><dt>Accepted</dt><dd id="acceptCount">0</dd></dl>
        <dl class="metric"><dt>Rejected / Errors</dt><dd id="rejectCount">0</dd></dl>
      </div>
      <div class="panel">
        <div class="toolbar">
          <div>
            <h2>Prediction Results</h2>
            <p>Click a row to review its original image and PatchCore overlay.</p>
          </div>
          <button id="exportCsv" class="secondary" type="button" disabled>Export CSV</button>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>File</th>
                <th>Verdict</th>
                <th>Score</th>
                <th>Threshold</th>
                <th>Decision</th>
              </tr>
            </thead>
            <tbody id="resultsBody">
              <tr><td colspan="5">No predictions yet.</td></tr>
            </tbody>
          </table>
        </div>
      </div>
      <div class="panel stack">
        <div>
          <h2>Selected Image Review</h2>
          <p id="selectedLabel">Select a scored image to inspect the heatmap overlay.</p>
        </div>
        <div class="media">
          <figure>
            <figcaption>Original</figcaption>
            <div id="originalWrap" class="placeholder">No image selected</div>
          </figure>
          <figure>
            <figcaption>PatchCore overlay</figcaption>
            <div id="overlayWrap" class="placeholder">No overlay selected</div>
          </figure>
        </div>
        <dl id="selectedDetails" class="details" hidden>
          <dt>Score</dt><dd id="detailScore"></dd>
          <dt>Threshold</dt><dd id="detailThreshold"></dd>
          <dt>Decision</dt><dd id="detailDecision"></dd>
        </dl>
      </div>
    </section>
  </main>
  <script>
    const supportedExtensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"];
    const drop = document.getElementById("drop");
    const fileInput = document.getElementById("file");
    const choose = document.getElementById("choose");
    const clearButton = document.getElementById("clear");
    const exportCsvButton = document.getElementById("exportCsv");
    const status = document.getElementById("status");
    const resultsBody = document.getElementById("resultsBody");
    const originalWrap = document.getElementById("originalWrap");
    const overlayWrap = document.getElementById("overlayWrap");
    const selectedLabel = document.getElementById("selectedLabel");
    const selectedDetails = document.getElementById("selectedDetails");
    const state = { rows: [], selectedId: null };

    choose.addEventListener("click", () => fileInput.click());
    clearButton.addEventListener("click", clearResults);
    exportCsvButton.addEventListener("click", exportCsv);
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) processFiles(Array.from(fileInput.files));
      fileInput.value = "";
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
      const files = Array.from(event.dataTransfer.files || []);
      if (files.length) processFiles(files);
    });

    async function processFiles(files) {
      choose.disabled = true;
      exportCsvButton.disabled = true;
      setStatus(`Scoring ${files.length} file${files.length === 1 ? "" : "s"}...`);
      if (!state.rows.length) resultsBody.innerHTML = "";
      for (const file of files) {
        const row = createPendingRow(file);
        state.rows.push(row);
        renderRows();
        selectRow(row.id);
        try {
          validateClientFile(file);
          const payload = await predict(file);
          Object.assign(row, {
            status: "done",
            filename: payload.filename,
            score: payload.score,
            threshold: payload.threshold,
            prediction: payload.prediction,
            decision: payload.decision,
            verdict: payload.verdict,
            warning: payload.warning,
            overlayUrl: `data:image/png;base64,${payload.overlay_png_base64}`,
            heatmapUrl: `data:image/png;base64,${payload.heatmap_png_base64}`
          });
        } catch (err) {
          Object.assign(row, {
            status: "error",
            verdict: "ERROR",
            decision: err.message,
            warning: err.message
          });
        }
        renderRows();
        selectRow(row.id);
      }
      updateCounts();
      choose.disabled = false;
      exportCsvButton.disabled = state.rows.length === 0;
      setStatus(`Finished ${files.length} file${files.length === 1 ? "" : "s"}.`);
    }

    function validateClientFile(file) {
      const lower = file.name.toLowerCase();
      if (!supportedExtensions.some((ext) => lower.endsWith(ext))) {
        throw new Error(`Unsupported file type. Use ${supportedExtensions.join(", ")}.`);
      }
    }

    async function predict(file) {
      const form = new FormData();
      form.append("image", file);
      const response = await fetch("/predict", { method: "POST", body: form });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Prediction failed");
      if (payload.warning) setStatus(payload.warning, "warn");
      return payload;
    }

    function createPendingRow(file) {
      return {
        id: crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`,
        filename: file.name,
        previewUrl: URL.createObjectURL(file),
        status: "pending",
        verdict: "SCORING",
        score: null,
        threshold: null,
        decision: "scoring",
        warning: null,
        prediction: null,
        overlayUrl: null
      };
    }

    function renderRows() {
      resultsBody.innerHTML = "";
      if (!state.rows.length) {
        resultsBody.innerHTML = '<tr><td colspan="5">No predictions yet.</td></tr>';
        updateCounts();
        return;
      }
      for (const row of state.rows) {
        const tr = document.createElement("tr");
        tr.className = "result-row";
        if (row.id === state.selectedId) tr.classList.add("selected");
        tr.addEventListener("click", () => selectRow(row.id));
        tr.innerHTML = `
          <td>${escapeHtml(row.filename)}</td>
          <td><span class="${badgeClass(row)}">${escapeHtml(shortVerdict(row))}</span></td>
          <td>${formatNumber(row.score)}</td>
          <td>${formatNumber(row.threshold)}</td>
          <td>${escapeHtml(row.warning || row.decision)}</td>
        `;
        resultsBody.appendChild(tr);
      }
      updateCounts();
    }

    function selectRow(id) {
      state.selectedId = id;
      const row = state.rows.find((item) => item.id === id);
      if (!row) return;
      renderRows();
      selectedLabel.textContent = row.filename;
      originalWrap.className = "";
      originalWrap.innerHTML = `<img alt="Original ${escapeHtml(row.filename)}" src="${row.previewUrl}" />`;
      overlayWrap.innerHTML = "";
      if (row.overlayUrl) {
        overlayWrap.className = "";
        overlayWrap.innerHTML = `<img alt="PatchCore overlay ${escapeHtml(row.filename)}" src="${row.overlayUrl}" />`;
      } else {
        overlayWrap.className = "placeholder";
        overlayWrap.textContent = row.status === "pending" ? "Scoring..." : "No overlay available";
      }
      selectedDetails.hidden = false;
      document.getElementById("detailScore").textContent = formatNumber(row.score);
      document.getElementById("detailThreshold").textContent = formatNumber(row.threshold);
      document.getElementById("detailDecision").textContent = row.warning || row.decision;
    }

    function updateCounts() {
      const total = state.rows.length;
      const accepted = state.rows.filter((row) => row.decision === "accept_valid_faf").length;
      const rejectedOrError = state.rows.filter(
        (row) => row.decision === "reject_ood_or_invalid" || row.status === "error"
      ).length;
      document.getElementById("totalCount").textContent = total;
      document.getElementById("acceptCount").textContent = accepted;
      document.getElementById("rejectCount").textContent = rejectedOrError;
    }

    function exportCsv() {
      if (!state.rows.length) return;
      const headers = ["filename", "verdict", "score", "threshold", "prediction", "decision", "warning"];
      const csv = [
        headers.join(","),
        ...state.rows.map((row) => headers.map((key) => csvCell(row[key])).join(","))
      ].join("\n");
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "gatekeeper_ui_predictions.csv";
      link.click();
      URL.revokeObjectURL(url);
    }

    function clearResults() {
      for (const row of state.rows) URL.revokeObjectURL(row.previewUrl);
      state.rows = [];
      state.selectedId = null;
      exportCsvButton.disabled = true;
      renderRows();
      selectedLabel.textContent = "Select a scored image to inspect the heatmap overlay.";
      selectedDetails.hidden = true;
      originalWrap.className = "placeholder";
      originalWrap.textContent = "No image selected";
      overlayWrap.className = "placeholder";
      overlayWrap.textContent = "No overlay selected";
      setStatus("Ready. Supported files: PNG, JPEG, TIFF, BMP.");
    }

    function badgeClass(row) {
      if (row.status === "error") return "badge error";
      if (row.decision === "accept_valid_faf") return "badge accept";
      if (row.decision === "reject_ood_or_invalid") return "badge reject";
      return "badge warn";
    }

    function shortVerdict(row) {
      if (row.status === "error") return "ERROR";
      if (row.decision === "accept_valid_faf") return "ACCEPT";
      if (row.decision === "reject_ood_or_invalid") return "REJECT";
      return row.verdict || "SCORING";
    }

    function formatNumber(value) {
      return value === null || value === undefined ? "not set" : Number(value).toFixed(6);
    }

    function setStatus(message, kind = "") {
      status.textContent = message;
      status.className = kind ? `status ${kind}` : "status";
    }

    function csvCell(value) {
      const text = value === null || value === undefined ? "" : String(value);
      return `"${text.replaceAll('"', '""')}"`;
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
