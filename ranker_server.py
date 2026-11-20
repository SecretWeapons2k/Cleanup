"""Tiny local server shared by build_ranker.py and build_gate_labeler.py.

A plain HTML file cannot write into a fixed folder. Served through this server
(http://127.0.0.1:PORT/), the pages save their CSV into ``ranker_out/`` next to the scripts
(manual export + automatic backup after every change). Opened straight from disk (file://)
the pages still work and fall back to the normal browser download.

Safety: binds to 127.0.0.1 only, accepts POST /save only for ``<safe-name>.csv``, writes only
inside the export folder and rejects requests whose Origin is not this server.
"""
from __future__ import annotations

import http.server
import json
import re
import socketserver
import threading
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DEFAULT_PORT = 8765

# Needs on the page: buildCsv() -> string, AUTOSAVE_NAME, RUN_ID, an element #saveState.
SAVE_JS = r"""
const SERVED = location.protocol.startsWith("http");
let _toastTimer = null, _autoTimer = null;
function toast(msg) {
  let t = document.getElementById("toast");
  if (!t) { t = document.createElement("div"); t.id = "toast";
    t.style.cssText = "position:fixed;right:14px;bottom:14px;background:#222;color:#fff;padding:8px 12px;border-radius:6px;font-size:13px;z-index:30;max-width:70vw;word-break:break-all"; document.body.appendChild(t); }
  t.textContent = msg; t.style.display = "block"; clearTimeout(_toastTimer); _toastTimer = setTimeout(() => { t.style.display = "none"; }, 6000);
}
async function saveToServer(name, text) {
  const r = await fetch("/save?name=" + encodeURIComponent(name), { method: "POST", body: text });
  if (!r.ok) throw new Error("HTTP " + r.status);
  return (await r.json()).path;
}
function downloadText(name, text) {
  const blob = new Blob([text], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
}
function scheduleAutosave() {
  const s = document.getElementById("saveState");
  if (!SERVED) { if (s) s.textContent = "nur Browser-Speicher (Seite nicht über den Server geöffnet)"; return; }
  clearTimeout(_autoTimer);
  _autoTimer = setTimeout(async () => {
    try { await saveToServer(AUTOSAVE_NAME, buildCsv()); if (s) s.textContent = "✓ gesichert in ranker_out"; }
    catch (e) { if (s) s.textContent = "⚠ Sicherung in ranker_out fehlgeschlagen"; }
  }, 700);
}
async function exportCsv(prefix) {
  const text = buildCsv();
  if (SERVED) {
    const d = new Date(), p2 = n => String(n).padStart(2, "0");
    const stamp = d.getFullYear() + p2(d.getMonth() + 1) + p2(d.getDate()) + "_" + p2(d.getHours()) + p2(d.getMinutes()) + p2(d.getSeconds());
    try { toast("Gespeichert: " + await saveToServer(prefix + "_" + RUN_ID + "_" + stamp + ".csv", text)); return; }
    catch (e) { toast("Server-Speichern fehlgeschlagen, lade stattdessen herunter"); }
  }
  downloadText(prefix + "_" + RUN_ID + ".csv", text);
}
"""


def _handler(run_dir: Path, out_dir: Path, port: int):
    allowed = {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(run_dir), **k)

        def log_message(self, *_a):  # keep the terminal quiet
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_POST(self):
            url = urlparse(self.path)
            origin = self.headers.get("Origin")
            if origin and origin not in allowed:
                self.send_error(403, "bad origin")
                return
            if url.path != "/save":
                self.send_error(404)
                return
            name = parse_qs(url.query).get("name", [""])[0]
            if not re.fullmatch(r"[A-Za-z0-9_.\-]{1,120}\.csv", name):
                self.send_error(400, "bad file name")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 20_000_000:
                self.send_error(413 if length > 0 else 400)
                return
            data = self.rfile.read(length)
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / name
            target.write_bytes(data)
            body = json.dumps({"path": str(target)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_server(run_dir: Path, out_dir: Path, port: int = DEFAULT_PORT) -> tuple[_Server, int]:
    """Bind the first free port from ``port`` on (up to +10). Returns (server, port)."""
    last_error: Exception | None = None
    for p in range(port, port + 11):
        try:
            return _Server(("127.0.0.1", p), _handler(run_dir, out_dir, p)), p
        except OSError as exc:
            last_error = exc
    raise SystemExit(f"Kein freier Port ab {port}: {last_error}")


def serve(run_dir: Path, out_dir: Path, page: str, port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    requested = port
    server, port = start_server(run_dir, out_dir, port)
    url = f"http://127.0.0.1:{port}/{page}"
    print(f"\nServer läuft: {url}")
    print(f"CSV-Dateien werden gespeichert in: {out_dir}")
    if port != requested:
        print(f"HINWEIS: Port {requested} war belegt, es wird {port} verwendet. "
              "Der Browser-Speicher gilt pro Adresse - Labels der anderen Adresse ggf. per 'CSV importieren' laden.")
    print("Beenden mit Strg+C.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer beendet.")
    finally:
        server.server_close()
