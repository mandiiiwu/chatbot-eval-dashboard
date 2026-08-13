"""Dispatches target-model chat calls to whichever provider is configured
(harness.ollama_client, or harness.custom_client for V2-G). evaluator.py
imports chat() from here instead of a specific provider's client directly,
so it stays provider-agnostic -- same domain-agnostic-code principle
already applied everywhere else in harness/ (see PLAN.md's design
principle at the top)."""

from . import config, custom_client, ollama_client


def chat(model: str, messages: list[dict]) -> str:
    if config.TARGET_PROVIDER == "custom":
        return custom_client.chat(model, messages)
    return ollama_client.chat(model, messages)
