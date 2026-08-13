"""Dispatches target-model chat calls to whichever provider is configured
(harness.ollama_client, or harness.custom_client for V2-G). evaluator.py
imports chat() from here instead of a specific provider's client directly,
so it stays provider-agnostic -- same domain-agnostic-code principle
already applied everywhere else in harness/ (see PLAN.md's design
principle at the top)."""

from . import config, custom_client, ollama_client


def chat(
    model: str,
    messages: list[dict],
    provider: str | None = None,
    endpoint_url: str | None = None,
    endpoint_headers: str | None = None,
    request_template: str | None = None,
    response_path: str | None = None,
) -> str:
    """provider and the endpoint_* params override config.TARGET_PROVIDER /
    config.CUSTOM_* for this call only -- lets the dashboard's [CONFIG]
    endpoint fields override .env per-run, same pattern as target_model."""
    provider = provider or config.TARGET_PROVIDER
    if provider == "custom":
        return custom_client.chat(
            model,
            messages,
            endpoint_url=endpoint_url,
            endpoint_headers=endpoint_headers,
            request_template=request_template,
            response_path=response_path,
        )
    return ollama_client.chat(model, messages)
