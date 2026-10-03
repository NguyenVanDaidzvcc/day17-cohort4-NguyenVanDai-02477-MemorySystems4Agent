from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from .model_provider import ProviderConfig, normalize_provider
except ImportError:
    from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    offline: bool = True
    profile_confidence_threshold: float = 0.85

    def __post_init__(self) -> None:
        if self.compact_threshold_tokens < 32:
            raise ValueError("COMPACT_THRESHOLD_TOKENS must be at least 32.")
        if self.compact_keep_messages < 1:
            raise ValueError("COMPACT_KEEP_MESSAGES must be positive.")
        if not 0 <= self.profile_confidence_threshold <= 1:
            raise ValueError("PROFILE_CONFIDENCE_THRESHOLD must be between 0 and 1.")


def _provider_config(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    provider = normalize_provider(os.getenv(f"{prefix}_PROVIDER", fallback.provider if fallback else "openai"))
    defaults = {
        "openai": "gpt-4o-mini", "custom": "local-model",
        "gemini": "gemini-2.5-flash", "anthropic": "claude-sonnet-4-5",
        "ollama": "llama3.2", "openrouter": "openai/gpt-4o-mini",
    }
    key_vars = {
        "openai": "OPENAI_API_KEY", "custom": "CUSTOM_API_KEY",
        "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY", "ollama": "OLLAMA_API_KEY",
    }
    same_provider = fallback is not None and fallback.provider == provider
    api_key = os.getenv(f"{prefix}_API_KEY") or os.getenv(key_vars[provider])
    if provider == "gemini":
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
    base_url = os.getenv(f"{prefix}_BASE_URL") or os.getenv(f"{provider.upper()}_BASE_URL")
    return ProviderConfig(
        provider=provider,
        model_name=os.getenv(f"{prefix}_MODEL", fallback.model_name if same_provider else defaults[provider]),
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", str(fallback.temperature if same_provider else 0.0))),
        api_key=api_key or (fallback.api_key if same_provider else None),
        base_url=base_url or (fallback.base_url if same_provider else None),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(root / ".env", override=False)
    state_dir = Path(os.getenv("STATE_DIR", "state"))
    if not state_dir.is_absolute():
        state_dir = root / state_dir
    model = _provider_config("LLM")
    mode = os.getenv("LAB_MODE", "offline").strip().lower()
    if mode not in {"offline", "live"}:
        raise ValueError("LAB_MODE must be offline or live.")
    config = LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir.resolve(),
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1200")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "6")),
        model=model,
        judge_model=_provider_config("JUDGE", model),
        offline=mode == "offline",
        profile_confidence_threshold=float(os.getenv("PROFILE_CONFIDENCE_THRESHOLD", "0.85")),
    )
    config.state_dir.mkdir(parents=True, exist_ok=True)
    return config
