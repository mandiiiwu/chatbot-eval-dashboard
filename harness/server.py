"""Minimal local server for the live dashboard (Phase 7, redesigned Phase 7-v2).
Regenerates the report fresh from current results/ state on every request to
`/`, so visiting localhost always shows the latest data—no manual re-run or
file-reopening step, no build tooling, no framework.

Also serves:
  - POST /run—triggers a real eval run (harness.evaluator.run_and_save—
    the same function run_eval.py's CLI uses), optionally overriding
    target_model and/or questions_file (a filename under questions/,
    validated via _safe_questions_path -- same path-traversal defense as
    corpus filenames) for this run only. Still a synchronous blocking POST,
    not a job queue or websockets -- but the server is now a
    ThreadingHTTPServer (not a plain single-threaded TCPServer) so a second
    request, namely the dashboard's progress poll below, can actually be
    answered while /run is still running instead of queuing behind it.
  - GET /run/progress—polled by the RUN_EVAL button while a run is in
    flight, so "RUNNING..." can show "RUNNING... 12/35" instead of sitting
    blank for however long a real model takes across 35 questions x 2
    calls each. Added after a real run got interrupted by an unrelated
    server restart and looked indistinguishable from "just hung" with no
    way to tell from the UI alone.
  - GET /run/cost_estimate—harness.evaluator.estimate_cost(), a rough
    (low, high) USD range computed from local data only (corpus chunk
    count, question/group count), no MicroDC calls involved. Fetched by
    the dashboard before RUN_EVAL's real POST fires, so the user sees "this
    will cost about ~$X-$Y, OK to proceed?" before any money is actually
    spent, not after.
  - POST /corpus/upload, POST /corpus/delete—lets the dashboard's corpus
    section actually add/remove corpus/*.md files instead of being a
    read-only display. Corpus files are always plain text, so uploads are
    JSON {filename, content}, not multipart form data; Python 3.13+
    dropped the stdlib cgi module that used to make multipart parsing easy,
    and text-only content sidesteps needing it at all.
  - POST /results/import—the dashboard's [IMPORT] button. Body is a raw
    results.json (harness.import_run.save() validates it strictly against
    the current schema before persisting it, same as a real RUN_EVAL run).
  - POST /corpus/upload_binary, POST /corpus/upload_url—harness.corpus_ingest's
    PDF/zip/URL-fetch support (see PLAN.md). Binary content (PDF/zip) arrives
    base64-encoded in JSON, same "no multipart parsing needed" reasoning as
    the plain-text upload path above.

A successful /run also syncs the daily scheduled job (harness.scheduling) to
whatever target_model was just used, and every request re-reads .env
(config.reload()) — both so this persistent, launchd-managed server always
reflects the current on-disk configuration without needing a manual restart.
"""

import base64
import binascii
import http.server
import json
import os
import threading
import urllib.parse

from . import config, corpus_ingest, import_run, report, retrieval, scheduling
from .evaluator import estimate_cost, run_and_save

PORT = int(os.environ.get("DASHBOARD_PORT", "8765"))

_CORPUS_EXTENSIONS = (".md", ".txt")

# Self-hosted font files (see report.py's @font-face rules) -- served from
# this fixed location rather than falling through to SimpleHTTPRequestHandler's
# default static-file handling, since serve() chdir()s into config.RESULTS_DIR
# for that fallback and these assets live alongside this module instead.
_STATIC_FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "fonts")

# Shared across request-handling threads now that the server is threaded --
# _progress_lock guards every read/write so a poll from one thread can't see
# a half-updated dict while /run's thread is mid-write, and so two /run
# clicks can't both start at once (checked-and-set under the same lock).
_progress_lock = threading.Lock()
# "phase" distinguishes the corpus-wide embedding phase from the
# per-question evaluation phase -- added 2026-08-16 after a real run
# against the 116,616-chunk CUAD corpus sat at "starting..." for hours with
# both current/total stuck at 0, since the old progress callback only ever
# covered the question loop and the corpus embed happens entirely before
# question 1 (see evaluator.run_and_save()'s progress_callback docstring).
_progress = {"running": False, "phase": "", "current": 0, "total": 0}


class CorpusPathError(ValueError):
    pass


def _safe_corpus_path(filename: str) -> str:
    """Validates filename and returns its absolute path inside
    config.CORPUS_DIR. Rejects anything that could escape the corpus
    directory (path separators, "..", absolute paths); a filename here
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


def _safe_questions_path(filename: str) -> str:
    """Same defense-in-depth as _safe_corpus_path(), for the dashboard's
    questions-file selector -- a filename here also comes straight from a
    browser request (POST /run's body, GET /run/cost_estimate's query
    string), so it needs the same path-traversal guard before being joined
    onto config.QUESTIONS_DIR."""
    if not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
        raise CorpusPathError(f"invalid filename: {filename!r}")
    if not filename.lower().endswith(".json"):
        raise CorpusPathError(f"filename must end in .json: {filename!r}")
    path = os.path.join(config.QUESTIONS_DIR, filename)
    questions_real = os.path.realpath(config.QUESTIONS_DIR)
    path_real = os.path.realpath(path)
    if os.path.commonpath([questions_real, path_real]) != questions_real:
        raise CorpusPathError(f"filename escapes questions/: {filename!r}")
    return path


def _write_ingested_files(ingested: list[tuple[str, str]]) -> list[str]:
    """Writes each (filename, text) pair from corpus_ingest to disk. Every
    filename still goes through _safe_corpus_path() here even though
    corpus_ingest already sanitizes to a basename with a safe extension—
    defense in depth, same principle as the original corpus-upload
    path-traversal fix, not trusting a single layer to be the only guard."""
    written = []
    for filename, text in ingested:
        path = _safe_corpus_path(filename)
        with open(path, "w") as f:
            f.write(text)
        written.append(filename)
    retrieval.clear_caches()  # corpus changed, stale embeddings/keyword-tokens must go
    return written


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    # http.server.BaseHTTPRequestHandler defaults protocol_version to
    # "HTTP/1.0" -- under that, Python's own parse_request() never honors
    # a browser's "Connection: keep-alive" (its close_connection logic
    # explicitly requires protocol_version >= "HTTP/1.1" on *both* sides,
    # see cpython's http/server.py), so this handler always closed the TCP
    # connection after every single response while never sending an
    # explicit "Connection: close" header to say so. Chrome (and most
    # browsers) send HTTP/1.1 requests assuming keep-alive by default and
    # pool connections for reuse; without an explicit signal that this
    # server won't honor that, a later request can get sent down a
    # connection the server already tore down, hanging indefinitely with
    # no error on either side -- fetch() has no default timeout, so this
    # surfaces as exactly "click a button, nothing ever happens," and
    # oddly "fixes itself" with DevTools' Network tab open (which changes
    # Chrome's own connection-pooling/caching behavior enough to usually
    # dodge the race). Setting HTTP/1.1 here makes keep-alive negotiation
    # actually correct instead of silently mismatched -- safe because
    # every response already sets a real Content-Length (_send_json/
    # _send_html below, and SimpleHTTPRequestHandler's static-file
    # handling), which is HTTP/1.1 keep-alive's only real prerequisite.
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path.startswith("/static/fonts/"):
            self._serve_font(self.path[len("/static/fonts/"):])
            return
        if self.path == "/run/progress":
            with _progress_lock:
                self._send_json(dict(_progress), 200)
            return
        if self.path.startswith("/run/cost_estimate"):
            try:
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                questions_filename = (query.get("questions_file") or [None])[0]
                questions_file = _safe_questions_path(questions_filename) if questions_filename else None
                # auto_generate_questions defaults to True on estimate_cost()
                # itself now (2026-08-16) -- not passed explicitly here so
                # this call site can't drift out of sync with that default.
                low, high, will_generate_questions = estimate_cost(questions_file=questions_file)
                self._send_json(
                    {"ok": True, "low": low, "high": high, "will_generate_questions": will_generate_questions}, 200
                )
            except CorpusPathError as e:
                self._send_json({"ok": False, "error": str(e)}, 400)
            except SystemExit as e:
                # config.require_questions_file() (via estimate_cost() ->
                # load_questions()) hard-blocks via SystemExit when no
                # questions file is configured yet -- a real, common case
                # now that render_onboarding_html() makes RUN_EVAL reachable
                # before any questions/*.json exists at all. SystemExit
                # isn't an Exception subclass, so it silently escaped the
                # generic handler below and crashed this request's thread
                # instead of returning a clean error (caught live while
                # verifying the onboarding page, 2026-08-16) -- same fix
                # already applied to POST /run for the same underlying
                # exception type.
                self._send_json({"ok": False, "error": str(e)}, 400)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
            return
        config.reload()  # picks up .env edits without needing to restart this persistent server
        if self.path in ("/", "/index.html"):
            latest = os.path.join(config.RESULTS_DIR, "latest.json")
            if not os.path.exists(latest):
                # Before a first run ever completes there's no run data to
                # show, but the dashboard's own config controls (corpus
                # upload, model endpoint, RUN_EVAL) are exactly what a
                # fresh user would reach for first -- render_onboarding_html()
                # exposes them here too (same [CONFIG] rail/JS as the full
                # dashboard) instead of leaving this a dead end that forces
                # a CLI run before the UI is usable at all. See report.py's
                # module docstring and PLAN.md's 2026-08-16 addition.
                self._send_html(report.render_onboarding_html(), 200)
                return
            with open(latest) as f:
                results = json.load(f)
            self._send_html(report.render_html(results), 200)
            return
        super().do_GET()

    def do_POST(self):
        config.reload()  # same as do_GET -- .env edits should apply without a server restart
        if self.path == "/run":
            with _progress_lock:
                if _progress["running"]:
                    self._send_json({"ok": False, "error": "a run is already in progress"}, 409)
                    return
                _progress.update(running=True, phase="", current=0, total=0)

            def _on_progress(phase: str, done: int, total: int) -> None:
                with _progress_lock:
                    _progress.update(phase=phase, current=done, total=total)

            body = self._read_json_body() or {}
            target_model = body.get("target_model") or None
            endpoint_config = body.get("endpoint_config") or None
            questions_filename = body.get("questions_file") or None
            try:
                questions_file = (
                    _safe_questions_path(questions_filename) if questions_filename else None
                )
            except CorpusPathError as e:
                with _progress_lock:
                    _progress["running"] = False
                self._send_json({"ok": False, "error": str(e)}, 400)
                return
            try:
                # auto_generate_questions defaults to True on run_and_save()
                # itself (2026-08-16): if no questions file exists yet,
                # generate one from the attached corpus instead of
                # hard-blocking -- the exact same default the CLI
                # (run_eval.py) now uses too, so a corpus+model set up
                # through this button, the CLI, or dropped directly onto
                # disk all converge on the same behavior. Not passed
                # explicitly here so this call site can't drift out of sync
                # with that shared default.
                results = run_and_save(
                    questions_file=questions_file,
                    target_model=target_model,
                    endpoint_config=endpoint_config,
                    progress_callback=_on_progress,
                )
                try:
                    # A one-off manual run also becomes the ongoing daily
                    # monitoring target, not just a single test. Failure
                    # here shouldn't mask a real, already-saved eval run's
                    # success -- scheduling sync is a secondary concern.
                    # questions_file can be None here even after a
                    # successful run (auto-generation resolved one that
                    # wasn't known ahead of time) -- reconstruct the real
                    # path actually used from what run_and_save() recorded.
                    used_questions_file = questions_file
                    if not used_questions_file and results.get("questions_file"):
                        used_questions_file = os.path.join(config.QUESTIONS_DIR, results["questions_file"])
                    scheduling.ensure_daily_job(results["target_model"], used_questions_file)
                except Exception:
                    pass
                self._send_json({"ok": True}, 200)
            except SystemExit as e:
                # coverage_check.require_coverage() hard-blocks via SystemExit
                # (correct for the CLI—kills the process); here it must
                # not take the server down with it, so it's reported back as
                # a normal error response instead.
                self._send_json({"ok": False, "error": str(e)}, 400)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
            finally:
                with _progress_lock:
                    _progress["running"] = False
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
                retrieval.clear_caches()  # corpus changed, stale embeddings/keyword-tokens must go
                self._send_json({"ok": True}, 200)
            except CorpusPathError as e:
                self._send_json({"ok": False, "error": str(e)}, 400)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
            return

        if self.path == "/corpus/upload_binary":
            body = self._read_json_body() or {}
            try:
                filename = body.get("filename", "")
                if not filename:
                    raise corpus_ingest.IngestError("no filename given")
                try:
                    data = base64.b64decode(body.get("content_base64", ""), validate=True)
                except binascii.Error as e:
                    raise corpus_ingest.IngestError(f"invalid base64 content: {e}")
                ingested = corpus_ingest.ingest_upload(filename, data)
                written = _write_ingested_files(ingested)
                self._send_json({"ok": True, "filenames": written}, 200)
            except (corpus_ingest.IngestError, CorpusPathError) as e:
                self._send_json({"ok": False, "error": str(e)}, 400)
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)
            return

        if self.path == "/corpus/upload_url":
            body = self._read_json_body() or {}
            try:
                url = body.get("url", "")
                if not url:
                    raise corpus_ingest.IngestError("no URL given")
                ingested = corpus_ingest.fetch_url(url)
                written = _write_ingested_files(ingested)
                self._send_json({"ok": True, "filenames": written}, 200)
            except (corpus_ingest.IngestError, CorpusPathError) as e:
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
                retrieval.clear_caches()
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

    def _serve_font(self, filename: str) -> None:
        """Serves a self-hosted font file from _STATIC_FONTS_DIR. Same
        path-traversal defense as _safe_corpus_path()/_safe_questions_path()
        -- filename comes straight from the request path."""
        if not filename or "/" in filename or "\\" in filename or not filename.lower().endswith(".woff2"):
            self.send_error(404)
            return
        path = os.path.join(_STATIC_FONTS_DIR, filename)
        fonts_real = os.path.realpath(_STATIC_FONTS_DIR)
        path_real = os.path.realpath(path)
        if os.path.commonpath([fonts_real, path_real]) != fonts_real or not os.path.isfile(path_real):
            self.send_error(404)
            return
        with open(path_real, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "font/woff2")
        self.send_header("Content-Length", str(len(data)))
        # Safe to cache aggressively and indefinitely, unlike the dashboard's
        # own HTML/JSON responses -- these files only change on a code
        # deploy (this server restarts on deploy anyway), never per-request.
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self._close_after_response()
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, body: str, status: int) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")  # never safe to cache; always reflects current on-disk state
        self._close_after_response()
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, data: dict, status: int) -> None:
        encoded = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self._close_after_response()
        self.end_headers()
        self.wfile.write(encoded)

    def _close_after_response(self) -> None:
        """Explicitly close the TCP connection after this response, and say
        so -- overrides HTTP/1.1's default keep-alive negotiation for every
        response this handler sends. Added 2026-08-16, replacing an earlier
        (kept, still correct) protocol_version = "HTTP/1.1" fix that turned
        out not to resolve a real, still-unexplained report: RUN_EVAL's
        /run/cost_estimate fetch taking 90 seconds to several minutes with
        DevTools closed, consistently reproducible, but instant with
        DevTools open, in incognito, over both "localhost" and "127.0.0.1",
        with the machine's own network confirmed healthy the whole time
        (direct curl to this exact server: instant; a fresh top-level
        navigation to the exact same URL: instant; general browsing on the
        same machine: normal speed). The one remaining, unexplained
        difference was specifically "fetch() issued by this page's own
        script" vs "a fresh navigation to the identical URL" -- the leading
        candidate left is some kind of connection-reuse/pooling
        interaction specific to that combination, in that browser's
        profile, that couldn't be pinned down further remotely. Rather than
        keep guessing at increasingly obscure causes, this removes
        persistent-connection reuse entirely for this server -- every
        response gets a deliberately fresh TCP connection next time,
        eliminating an entire category of connection-state bugs outright
        instead of trying to out-guess one. The performance cost is
        negligible for a local, single-user tool (loopback connection
        setup measured at a fraction of a millisecond)."""
        self.send_header("Connection", "close")
        self.close_connection = True

    def log_message(self, format, *args):
        pass  # keep stdout quiet; this is a local dev tool, not a service


def serve() -> None:
    os.chdir(config.RESULTS_DIR)
    # ThreadingHTTPServer, not a plain single-threaded HTTPServer/TCPServer:
    # /run blocks for however long a real eval takes (minutes), and without
    # threading no other request -- including the progress poll this exists
    # to serve -- could be answered until it finished, defeating the point.
    # allow_reuse_address: without this, restarting the server (e.g. after
    # editing report.py) can fail with "Address already in use" for up to a
    # minute; the just-killed process's socket lingers in TIME_WAIT and
    # blocks rebinding even though no process is actually holding it.
    http.server.ThreadingHTTPServer.allow_reuse_address = True
    http.server.ThreadingHTTPServer.daemon_threads = True
    with http.server.ThreadingHTTPServer(("127.0.0.1", PORT), DashboardHandler) as httpd:
        print(f"Dashboard live at http://localhost:{PORT}  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    serve()
