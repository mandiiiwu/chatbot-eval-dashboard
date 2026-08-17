"""Converts corpus material that doesn't arrive as clean .md/.txt into plain
text the harness can use -- PDF text extraction, a few specific JSON
shapes, zip archives of mixed files, and fetching a file directly from a
URL. Deliberately NOT a per-platform dataset integration (Hugging Face,
Kaggle, etc): those wrap arbitrary, dataset-specific schemas, so
"pointing" at one wouldn't actually skip the hard part of turning it into
real reference text, and would repeat the "provider registry" mistake
already rejected once for V2-G's custom endpoint. JSON support here is
narrower on purpose, for the same reason -- only a few well-known,
widely-reused shapes are recognized (see extract_json_text), not arbitrary
JSON, since there's no universal "this is the reference text" convention
for JSON the way there is for a PDF's text layer. A generic raw-URL fetch
covers the case that generalizes (anything with a direct-download link)
without per-service special-casing. See PLAN.md for the full design note.
"""

import io
import json
import os
import zipfile

import requests
from pypdf import PdfReader

_SUPPORTED_EXTENSIONS = (".md", ".txt", ".pdf", ".json")
_MAX_FETCH_BYTES = 20 * 1024 * 1024  # sanity cap; a mistyped URL shouldn't hang the server


class IngestError(ValueError):
    pass


def extract_pdf_text(data: bytes) -> str:
    """Pulls the embedded text layer out of a PDF via pypdf (pure Python,
    MIT-licensed, no native build step -- avoids repeating the
    scispacy/thinc/blis compile-from-source problem already hit once in
    fact_check.py). OCR is out of scope: a scanned-image PDF with no text
    layer raises a clear error instead of silently returning empty content."""
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as e:
        raise IngestError(f"could not read this PDF: {e}")
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text.strip():
        raise IngestError(
            "no extractable text found in this PDF -- it may be a scanned image "
            "with no embedded text layer (OCR isn't supported)"
        )
    return text


def extract_json_text(data: bytes) -> str:
    """Pulls prose out of a JSON file, but only for a few specific,
    widely-used shapes -- not arbitrary JSON. There's no universal
    "this is the reference text" convention for JSON the way a PDF always
    has an extractable text layer, so an unrecognized shape is rejected
    with a clear error instead of guessing at one. Recognizes, in order:

      1. A flat array of strings.
      2. An array of objects each with a "text" field.
      3. SQuAD-format (CUAD and plenty of other QA datasets use this same
         shape): data[].paragraphs[].context. Only "context" is pulled --
         the paired "qas" (questions/answers) are deliberately skipped,
         since this harness generates its own questions from the corpus
         and importing someone else's premade Q&A as if it were reference
         prose would mean generating new questions about existing
         questions, plus reintroducing exactly the kind of
         not-directly-verified content the corpus rules already guard
         against for citations/headers."""
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as e:
        raise IngestError(f"not valid JSON: {e}")

    if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
        chunks = parsed
    elif isinstance(parsed, list) and all(isinstance(x, dict) and isinstance(x.get("text"), str) for x in parsed):
        chunks = [x["text"] for x in parsed]
    elif isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
        chunks = [
            para["context"]
            for entry in parsed["data"]
            for para in entry.get("paragraphs", [])
            if isinstance(para.get("context"), str)
        ]
        if not chunks:
            raise IngestError('JSON has a "data" array but no data[].paragraphs[].context strings found')
    else:
        raise IngestError(
            "unrecognized JSON shape -- supported: a flat array of strings, an "
            'array of {"text": ...} objects, or SQuAD-format '
            "(data[].paragraphs[].context)"
        )

    text = "\n\n".join(c.strip() for c in chunks if c and c.strip())
    if not text:
        raise IngestError("JSON matched a supported shape but contained no non-empty text")
    return text


def _corpus_filename_for(original: str) -> str:
    """PDF/JSON sources become .md (the extracted text is prose, same as
    any other corpus file); .txt/.md sources keep their own extension."""
    base = os.path.basename(original)
    root, ext = os.path.splitext(base)
    if ext.lower() in (".pdf", ".json"):
        return root + ".md"
    return base


def ingest_bytes(filename: str, data: bytes) -> str:
    """Given a filename (used only for its extension) and raw bytes,
    returns plain text ready to save as a corpus file. Raises IngestError
    for unsupported types. Handles exactly one file; see ingest_upload()
    for the entry point that also handles zip archives (many files)."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".md", ".txt"):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise IngestError(f"{filename}: not valid UTF-8 text ({e})")
    if ext == ".pdf":
        return extract_pdf_text(data)
    if ext == ".json":
        return extract_json_text(data)
    raise IngestError(f"{filename}: unsupported file type (only .md, .txt, .pdf, .json are supported)")


def ingest_zip(data: bytes) -> list[tuple[str, str]]:
    """Extracts .md/.txt/.pdf/.json members from a zip archive and ingests
    each to plain text. Returns [(corpus_filename, text), ...] for members that
    succeeded; silently skips directories, unsupported types, zip-internal
    junk (__MACOSX/, dotfiles), and any member that fails to extract,
    rather than failing the whole batch over one bad file -- a zip is
    expected to be a mixed folder dump, not every member is meant to
    become a corpus file.

    Only `os.path.basename()` of each member name is ever used, which
    defuses zip-slip path traversal (a member named e.g.
    "../../etc/passwd") before it reaches disk -- server.py additionally
    routes every resulting filename through _safe_corpus_path() as
    defense in depth, same pattern as the original corpus-upload fix."""
    results: list[tuple[str, str]] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise IngestError(f"not a valid zip file: {e}")
    with zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            basename = os.path.basename(info.filename)
            if not basename or basename.startswith(".") or "__MACOSX" in info.filename:
                continue
            ext = os.path.splitext(basename)[1].lower()
            if ext not in _SUPPORTED_EXTENSIONS:
                continue
            try:
                text = ingest_bytes(basename, zf.read(info))
            except IngestError:
                continue
            if text.strip():
                results.append((_corpus_filename_for(basename), text))
    if not results:
        raise IngestError("no usable .md/.txt/.pdf/.json files found in this zip")
    return results


def ingest_upload(filename: str, data: bytes) -> list[tuple[str, str]]:
    """Single entry point for turning an uploaded or fetched file into one
    or more (corpus_filename, text) pairs -- a .zip can produce many files,
    anything else produces exactly one. This is what server.py's upload
    and URL-fetch endpoints both call, so a zip behaves identically
    whether it arrived via direct upload or a URL fetch."""
    if os.path.splitext(filename)[1].lower() == ".zip":
        return ingest_zip(data)
    return [(_corpus_filename_for(filename), ingest_bytes(filename, data))]


def fetch_url(url: str) -> list[tuple[str, str]]:
    """Fetches a file directly from a URL and ingests it -- a .zip URL
    works too, same as an uploaded zip. Deliberately a plain HTTP GET, not
    a per-platform dataset integration -- see module docstring. Scheme
    restricted to http/https (rejects file:// etc -- cheap, prevents an
    obviously-wrong paste; not a full SSRF threat-model exercise, since
    this is a local single-user tool where the URL comes from the person
    running their own dashboard, not an untrusted third party)."""
    if not url.lower().startswith(("http://", "https://")):
        raise IngestError("only http:// and https:// URLs are supported")

    try:
        resp = requests.get(url, timeout=30, stream=True)
    except requests.exceptions.RequestException as e:
        raise IngestError(f"could not fetch URL: {e}")

    if resp.status_code != 200:
        raise IngestError(f"URL returned HTTP {resp.status_code}")

    content_length = resp.headers.get("Content-Length")
    if content_length and int(content_length) > _MAX_FETCH_BYTES:
        raise IngestError(f"file is too large ({int(content_length):,} bytes, max 20MB)")

    data = bytearray()
    for chunk in resp.iter_content(chunk_size=65536):
        data.extend(chunk)
        if len(data) > _MAX_FETCH_BYTES:
            raise IngestError("file is too large (exceeded 20MB while downloading)")

    filename = os.path.basename(url.split("?")[0].split("#")[0]) or "downloaded"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (*_SUPPORTED_EXTENSIONS, ".zip"):
        content_type = resp.headers.get("Content-Type", "")
        if "zip" in content_type:
            filename += ".zip"
        elif "pdf" in content_type:
            filename += ".pdf"
        elif "json" in content_type:
            filename += ".json"
        elif "text" in content_type or "markdown" in content_type:
            filename += ".txt"
        else:
            raise IngestError(
                f"couldn't tell what kind of file this is from the URL or "
                f"Content-Type ({content_type!r}) -- only .md, .txt, .pdf, .json, .zip are supported"
            )

    return ingest_upload(filename, bytes(data))
