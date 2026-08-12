"""Thin wrapper around MicroDC.ai's OpenAI-compatible chat completions
endpoint. No SDK needed -- it's a plain REST call."""

import time

import requests

from . import config


class MicroDCError(RuntimeError):
    pass


def chat(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 800,
    retries: int = 3,
) -> str:
    """Send a chat completion request, return the assistant's text reply.

    Disables extended thinking by default: some MicroDC models (e.g.
    gpt-oss:20b) support a reasoning mode that burns far more tokens and
    requires max_tokens >= 4096. We don't need chain-of-thought for short
    factual answers, so keep it off to keep runs fast and cheap.

    Retries on timeouts/connection errors: MicroDC runs jobs on a
    peer-to-peer GPU marketplace, so an individual worker occasionally
    stalls -- that's not a real failure, just retry against (likely) a
    different worker.
    """
    api_key = config.require_api_key()
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": False,
    }

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{config.MDC_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=120,
            )
        except requests.exceptions.RequestException as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
            continue

        if resp.status_code != 200:
            raise MicroDCError(f"MicroDC request failed ({resp.status_code}): {resp.text[:500]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise MicroDCError(f"MicroDC request timed out after {retries} attempts: {last_error}")
