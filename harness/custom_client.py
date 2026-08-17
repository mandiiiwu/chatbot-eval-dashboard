"""V2-G: generic templated HTTP client for any REST/JSON chatbot API that
isn't OpenAI-compatible (ollama_client.py already covers those). Covers a
custom in-house API's own request/response shape via user-supplied
templates, rather than one-off provider code—see config.py's comment
for why a provider registry doesn't actually generalize to "most
specific-purpose chatbots" the way a template does."""

import json
import time

import requests

from . import config


class CustomClientError(RuntimeError):
    pass


def _substitute(obj, replacements: dict[str, str]):
    """Recursively replaces placeholder strings within a *parsed* JSON
    structure's string values, not the raw template text; substituting
    before re-serializing means json.dumps() handles all escaping
    correctly no matter what characters the message/system text contains
    (quotes, newlines, etc). Templating the raw JSON string directly would
    risk the same class of bug already caught once this session in
    alerting.py's AppleScript quoting."""
    if isinstance(obj, str):
        for placeholder, value in replacements.items():
            obj = obj.replace(placeholder, value)
        return obj
    if isinstance(obj, dict):
        return {k: _substitute(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(v, replacements) for v in obj]
    return obj


def _extract(data, path: str):
    current = data
    for part in path.split("."):
        try:
            if isinstance(current, list):
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current[part]
            else:
                raise KeyError(part)
        except (KeyError, IndexError, ValueError) as e:
            raise CustomClientError(
                f"CUSTOM_RESPONSE_PATH {path!r} failed at segment {part!r} ({e}). "
                f"Actual response: {json.dumps(data)[:300]}"
            )
    return current


def chat(
    model: str,
    messages: list[dict],
    retries: int = 2,
    endpoint_url: str | None = None,
    endpoint_headers: str | None = None,
    request_template: str | None = None,
    response_path: str | None = None,
) -> str:
    """Same call shape as ollama_client.chat() so harness/target_client.py
    can dispatch between providers without evaluator.py knowing which one
    is active. `model` becomes the {{model}} template placeholder—a
    label, not necessarily a real field the target API needs; only used if
    the user's own CUSTOM_REQUEST_TEMPLATE references it.

    The four endpoint_* parameters are per-call overrides of the matching
    config.CUSTOM_* value—lets the dashboard's [CONFIG] endpoint fields
    override .env for a single run without mutating global config, same
    pattern already used for target_model (evaluator.run_evaluation).
    None (the default) means "use the .env value", not "use an empty one"."""
    url = endpoint_url or config.CUSTOM_ENDPOINT_URL
    headers_raw = endpoint_headers if endpoint_headers is not None else config.CUSTOM_ENDPOINT_HEADERS
    template_raw = request_template if request_template is not None else config.CUSTOM_REQUEST_TEMPLATE
    path = response_path or config.CUSTOM_RESPONSE_PATH

    if not url:
        raise CustomClientError(
            "No custom endpoint URL configured; set CUSTOM_ENDPOINT_URL in .env, "
            "or fill in the endpoint_url field on the dashboard."
        )

    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")

    try:
        template = json.loads(template_raw)
    except json.JSONDecodeError as e:
        raise CustomClientError(f"request_template isn't valid JSON: {e}")
    try:
        headers = json.loads(headers_raw)
    except json.JSONDecodeError as e:
        raise CustomClientError(f"endpoint_headers isn't valid JSON: {e}")

    payload = _substitute(template, {"{{model}}": model, "{{system}}": system, "{{message}}": user})

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=180)
        except requests.exceptions.RequestException as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
            continue

        if resp.status_code != 200:
            raise CustomClientError(f"Custom endpoint request failed ({resp.status_code}): {resp.text[:500]}")
        try:
            data = resp.json()
        except json.JSONDecodeError:
            raise CustomClientError(f"Custom endpoint returned non-JSON response: {resp.text[:300]}")
        return str(_extract(data, path))

    raise CustomClientError(f"Custom endpoint request failed after {retries} attempts: {last_error}")
