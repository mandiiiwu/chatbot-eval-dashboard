"""Minimal local server for the live dashboard (Phase 7, redesigned Phase 7-v2).
Regenerates the report fresh from current results/ state on every request to
`/`, so visiting localhost always shows the latest data -- no manual re-run or
file-reopening step, no build tooling, no framework.

Also serves POST /run, which triggers a real eval run (harness.evaluator.
run_and_save -- the same function run_eval.py's CLI uses) so the dashboard's
RUN_EVAL button works for real instead of being a non-functional mock. This
is a local, single-user tool, so a synchronous blocking POST is the right
level of complexity -- no job queue, no websockets, no polling."""

import http.server
import json
import os
import socketserver

from . import config, report
from .evaluator import run_and_save

PORT = int(os.environ.get("DASHBOARD_PORT", "8765"))


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
            try:
                run_and_save()
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
        self.send_error(404)

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
    with socketserver.TCPServer(("127.0.0.1", PORT), DashboardHandler) as httpd:
        print(f"Dashboard live at http://localhost:{PORT}  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    serve()
