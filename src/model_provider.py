from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    aliases = {
        "anthorpic": "anthropic", "claude": "anthropic",
        "google": "gemini", "google_genai": "gemini",
        "openai-compatible": "custom", "openai_compatible": "custom",
        "open-router": "openrouter",
    }
    provider = value.strip().lower()
    provider = aliases.get(provider, provider)
    if provider not in {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}:
        raise ValueError(f"Unsupported LLM provider: {value!r}")
    return provider


def build_chat_model(config: ProviderConfig) -> Any:
    """Lazy imports keep the offline lab independent of provider SDKs."""
    provider = normalize_provider(config.provider)
    integrations = {
        "openai": ("langchain_openai", "ChatOpenAI"),
        "custom": ("langchain_openai", "ChatOpenAI"),
        "gemini": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
        "anthropic": ("langchain_anthropic", "ChatAnthropic"),
        "ollama": ("langchain_ollama", "ChatOllama"),
        "openrouter": ("langchain_openrouter", "ChatOpenRouter"),
    }
    if provider == "custom" and not config.base_url:
        raise ValueError("CUSTOM_BASE_URL is required for the custom provider.")
    if provider not in {"custom", "ollama"} and not config.api_key:
        raise ValueError(f"An API key is required for live provider {provider}.")
    module, class_name = integrations[provider]
    try:
        model_class = getattr(import_module(module), class_name)
    except ImportError as exc:
        raise RuntimeError(f"Install {module.replace('_', '-')} to use {provider} live.") from exc
    kwargs: dict[str, Any] = {"model": config.model_name, "temperature": config.temperature}
    if provider != "ollama":
        kwargs.update(api_key=config.api_key or "local-no-key", timeout=60, max_retries=2)
    if config.base_url:
        if provider == "gemini":
            kwargs["client_options"] = {"api_endpoint": config.base_url}
        elif provider == "anthropic":
            kwargs["anthropic_api_url"] = config.base_url
        else:
            kwargs["base_url"] = config.base_url
    return model_class(**kwargs)
