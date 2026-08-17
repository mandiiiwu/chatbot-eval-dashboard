"""Thin wrapper around a local Ollama server's OpenAI-compatible chat
completions endpoint. Used only for the target model; the judge model
stays on MicroDC (see harness/microdc_client.py) to avoid a same-family
judge and to keep judge calls off the local machine's GPU/CPU."""

import time

import requests

from . import config


class OllamaError(RuntimeError):
    pass


def chat(
    model: str,
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 800,
    retries: int = 2,
) -> str:
    """Send a chat completion request to local Ollama, return the assistant's
    text reply. Retries on connection errors (e.g. the model is still being
    loaded into memory on the first call)."""
    if not model:
        # Validates the actual model that will be used for this call, not
        # config.TARGET_MODEL directly; V2-C's --target-model override
        # (run_eval.py) can supply a model without ever setting .env's
        # TARGET_MODEL, so checking the global would reject a valid call.
        raise SystemExit(
            "No target model specified. Set TARGET_MODEL in .env, or pass "
            "--target-model to run_eval.py (see `ollama list` for what's "
            "available locally)."
        )
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{config.OLLAMA_BASE_URL}/chat/completions",
                json=payload,
                timeout=180,
            )
        except requests.exceptions.RequestException as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
            continue

        if resp.status_code != 200:
            raise OllamaError(f"Ollama request failed ({resp.status_code}): {resp.text[:500]}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise OllamaError(
        f"Ollama request failed after {retries} attempts: {last_error}. "
        "Is Ollama running? (`brew services start ollama` or `ollama serve`)"
    )
