"""Application configuration loaded from environment variables / .env file."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the proxy server."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Server
    host: str = "127.0.0.1"
    port: int = 8082

    # Upstream OpenAI-compatible backend
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""

    # Model mapping (Anthropic tier -> OpenAI model)
    big_model: str = "gpt-4o"
    small_model: str = "gpt-4o-mini"

    # Optional behaviour
    request_timeout: int = 120
    max_tokens_limit: int = 0
    # If set, incoming requests must present this key via x-api-key / Authorization
    anthropic_api_key: str = ""

    @property
    def base_url(self) -> str:
        """Normalised upstream base URL without trailing slash."""
        return self.openai_base_url.rstrip("/")

    def map_model(self, anthropic_model: str) -> str:
        """Map an incoming Anthropic model name to a configured OpenAI model.

        Claude Code mainly uses the sonnet/opus (big) and haiku (small) tiers.
        Any name already looking like a non-claude model is passed through.
        """
        name = (anthropic_model or "").lower()
        if "haiku" in name:
            return self.small_model
        if "sonnet" in name or "opus" in name or "claude" in name:
            return self.big_model
        # Not a Claude model name -> assume caller passed a concrete model id.
        return anthropic_model


settings = Settings()
