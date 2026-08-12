"""Minimal local server for the live dashboard (Phase 7). Regenerates the
report fresh from current results/ state on every request to `/`, so
visiting localhost always shows the latest data -- no manual re-run or
file-reopening step, no build tooling, no framework, no websockets (a page
refresh is enough given the eval runs once a day via launchd)."""

import http.server
import json
import os
import socketserver

from . import config, report

PORT = int(os.environ.get("DASHBOARD_PORT", "8765"))


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            latest = os.path.join(config.RESULTS_DIR, "latest.json")
            if not os.path.exists(latest):
                self._send_html(
                    "<h1>No eval runs yet</h1><p>Run <code>python run_eval.py</code> first.</p>", 200
                )
                return
            with open(latest) as f:
                results = json.load(f)
            self._send_html(report.render_html(results), 200)
            return
        super().do_GET()

    def _send_html(self, body: str, status: int) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
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
