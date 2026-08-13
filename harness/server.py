"""Minimal local server for the live dashboard (Phase 7, redesigned Phase 7-v2).
Regenerates the report fresh from current results/ state on every request to
`/`, so visiting localhost always shows the latest data -- no manual re-run or
file-reopening step, no build tooling, no framework.

Also serves:
  - POST /run -- triggers a real eval run (harness.evaluator.run_and_save --
    the same function run_eval.py's CLI uses), optionally overriding
    target_model for this run only. Local, single-user tool, so a
    synchronous blocking POST is the right level of complexity -- no job
    queue, no websockets, no polling.
  - POST /corpus/upload, POST /corpus/delete -- lets the dashboard's corpus
    section actually add/remove corpus/*.md files instead of being a
    read-only display. Corpus files are always plain text, so uploads are
    JSON {filename, content}, not multipart form data -- Python 3.13+
    dropped the stdlib cgi module that used to make multipart parsing easy,
    and text-only content sidesteps needing it at all.
  - POST /results/import -- the dashboard's [IMPORT] button. Body is a raw
    results.json (harness.import_run.save() validates it strictly against
    the current schema before persisting it, same as a real RUN_EVAL run).
"""

import http.server
import json
import os
import socketserver

from . import config, import_run, report, retrieval
from .evaluator import run_and_save

PORT = int(os.environ.get("DASHBOARD_PORT", "8765"))

_CORPUS_EXTENSIONS = (".md", ".txt")


class CorpusPathError(ValueError):
    pass


def _safe_corpus_path(filename: str) -> str:
    """Validates filename and returns its absolute path inside
    config.CORPUS_DIR. Rejects anything that could escape the corpus
    directory (path separators, "..", absolute paths) -- a filename here
    comes straight from a browser request, and blindly os.path.join-ing
    user input is a classic path-traversal hole even on a single-user
    localhost tool. Also requires a .md/.txt extension, matching what
    retrieval.py already treats as corpus content."""
    if not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
        raise CorpusPathError(f"invalid filename: {filename!r}")
    if not filename.lower().endswith(_CORPUS_EXTENSIONS):
        raise CorpusPathError(f"filename must end in .md or .txt: {filename!r}")
    path = os.path.join(config.CORPUS_DIR, filename)
    # Defense in depth: even after the checks above, confirm the resolved
    # real path is still actually inside CORPUS_DIR before touching disk.
    corpus_real = os.path.realpath(config.CORPUS_DIR)
    path_real = os.path.realpath(path)
    if os.path.commonpath([corpus_real, path_real]) != corpus_real:
        raise CorpusPathError(f"filename escapes corpus/: {filename!r}")
    return path


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            latest = os.path.join(config.RESULTS_DIR, "latest.json")
            if not os.path.exists(latest):
                self._send_html(
                    "<h1>No eval runs yet</h1><p>Run <code>python run_eval.py</code> "
                    "or click RUN_EVAL once a target model is configured.</p>", 200
                )
                return
            with open(latest) as f:
                results = json.load(f)
            self._send_html(report.render_html(results), 200)
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/run":
            body = self._read_json_body() or {}
            target_model = body.get("target_model") or None
            endpoint_config = body.get("endpoint_config") or None
            try:
                run_and_save(target_model=target_model, endpoint_config=endpoint_config)
                self._send_json({"ok": True}, 200)
            except SystemExit as e:
                # coverage_check.require_coverage() hard-blocks via SystemExit
                # (correct for the CLI -- kills the process); here it must
                # not take the server down with it, so it's reported back as
                # a normal error response instead.
                self._send_json({"ok": False, "error": str(e)}, 400)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
            return

        if self.path == "/corpus/upload":
            body = self._read_json_body() or {}
            try:
                path = _safe_corpus_path(body.get("filename", ""))
                content = body.get("content", "")
                if not content.strip():
                    raise CorpusPathError("file is empty")
                with open(path, "w") as f:
                    f.write(content)
                retrieval._chunk_embeddings.cache_clear()  # corpus changed, stale embeddings must go
                self._send_json({"ok": True}, 200)
            except CorpusPathError as e:
                self._send_json({"ok": False, "error": str(e)}, 400)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
            return

        if self.path == "/results/import":
            body = self._read_json_body()
            try:
                if body is None:
                    raise import_run.ImportValidationError("request body is empty or not valid JSON")
                filename = import_run.save(body)
                self._send_json({"ok": True, "filename": filename}, 200)
            except import_run.ImportValidationError as e:
                self._send_json({"ok": False, "error": str(e)}, 400)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
            return

        if self.path == "/corpus/delete":
            body = self._read_json_body() or {}
            try:
                path = _safe_corpus_path(body.get("filename", ""))
                if not os.path.exists(path):
                    raise CorpusPathError(f"no such corpus file: {body.get('filename')!r}")
                os.remove(path)
                retrieval._chunk_embeddings.cache_clear()
                self._send_json({"ok": True}, 200)
            except CorpusPathError as e:
                self._send_json({"ok": False, "error": str(e)}, 400)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
            return

        self.send_error(404)

    def _read_json_body(self) -> dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _send_html(self, body: str, status: int) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, data: dict, status: int) -> None:
        encoded = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        pass  # keep stdout quiet -- this is a local dev tool, not a service


def serve() -> None:
    os.chdir(config.RESULTS_DIR)
    # allow_reuse_address: without this, restarting the server (e.g. after
    # editing report.py) can fail with "Address already in use" for up to a
    # minute -- the just-killed process's socket lingers in TIME_WAIT and
    # blocks rebinding even though no process is actually holding it.
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), DashboardHandler) as httpd:
        print(f"Dashboard live at http://localhost:{PORT}  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    serve()
