"""Runtime-mutable configuration.

Unlike :mod:`app.config` (static server settings read from the environment at
startup), this holds the values a user edits from the web setup page:

* the upstream OpenAI-compatible base URL + API key
* the big / small model mapping
* the auto-generated Anthropic API key that Claude Code must present

Changes are persisted to ``config.json`` so they survive restarts and take
effect immediately for subsequent requests.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
from pathlib import Path
from typing import Any, Dict

from .config import settings

_CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", "config.json"))


def _generate_anthropic_key() -> str:
    """Create a stable-looking Anthropic-style key for local use."""
    return "sk-ant-proxy-" + secrets.token_hex(24)


def _mask(value: str) -> str:
    """Mask a secret for display, keeping only a short prefix/suffix."""
    if not value:
        return ""
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


class RuntimeConfig:
    """Thread-safe, persistable runtime configuration."""

    _EDITABLE = ("openai_base_url", "openai_api_key", "big_model", "small_model")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # seed from environment-based defaults
        self.openai_base_url = settings.openai_base_url
        self.openai_api_key = settings.openai_api_key
        self.big_model = settings.big_model
        self.small_model = settings.small_model
        self.anthropic_api_key = settings.anthropic_api_key
        self._load()
        if not self.anthropic_api_key:
            self.anthropic_api_key = _generate_anthropic_key()
            self._save()

    # --- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not _CONFIG_PATH.exists():
            return
        try:
            data = json.loads(_CONFIG_PATH.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for key in (*self._EDITABLE, "anthropic_api_key"):
            if data.get(key):
                setattr(self, key, data[key])

    def _save(self) -> None:
        payload = {
            "openai_base_url": self.openai_base_url,
            "openai_api_key": self.openai_api_key,
            "big_model": self.big_model,
            "small_model": self.small_model,
            "anthropic_api_key": self.anthropic_api_key,
        }
        try:
            _CONFIG_PATH.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), "utf-8"
            )
        except OSError:
            pass

    # --- derived -----------------------------------------------------------
    @property
    def base_url(self) -> str:
        """Normalised upstream base URL without trailing slash."""
        return self.openai_base_url.rstrip("/")

    def anthropic_base_url(self) -> str:
        """The address Claude Code should point ANTHROPIC_BASE_URL at."""
        host = settings.host if settings.host not in ("0.0.0.0",) else "127.0.0.1"
        return f"http://{host}:{settings.port}"

    def map_model(self, anthropic_model: str) -> str:
        """Map an incoming Anthropic model name to the configured model."""
        name = (anthropic_model or "").lower()
        if "haiku" in name:
            return self.small_model
        if "sonnet" in name or "opus" in name or "claude" in name:
            return self.big_model
        return anthropic_model

    def is_ready(self) -> bool:
        """Whether upstream is configured well enough to proxy requests."""
        return bool(self.base_url) and bool(self.openai_api_key)

    # --- mutation ----------------------------------------------------------
    def update(self, values: Dict[str, Any]) -> None:
        """Apply editable fields. Empty strings are ignored (keep existing)."""
        with self._lock:
            for key in self._EDITABLE:
                if key not in values:
                    continue
                val = values[key]
                if val is None:
                    continue
                if isinstance(val, str):
                    val = val.strip()
                # An empty API key means "leave unchanged" (form left blank).
                if key == "openai_api_key" and val == "":
                    continue
                if val == "":
                    continue
                setattr(self, key, val)
            self._save()

    def regenerate_anthropic_key(self) -> str:
        with self._lock:
            self.anthropic_api_key = _generate_anthropic_key()
            self._save()
            return self.anthropic_api_key

    # --- views -------------------------------------------------------------
    def public_dict(self) -> Dict[str, Any]:
        """Config for the setup page (upstream key is masked)."""
        return {
            "openai_base_url": self.openai_base_url,
            "openai_api_key_masked": _mask(self.openai_api_key),
            "openai_api_key_set": bool(self.openai_api_key),
            "big_model": self.big_model,
            "small_model": self.small_model,
            "anthropic_base_url": self.anthropic_base_url(),
            "anthropic_api_key": self.anthropic_api_key,
            "ready": self.is_ready(),
        }


config = RuntimeConfig()
