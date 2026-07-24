"""Convenience entry point: ``python server.py`` starts the proxy."""
from __future__ import annotations

import uvicorn

from app.config import settings
from app.runtime_config import config


def main() -> None:
    print("=" * 64)
    print(" OpenAI → Anthropic Proxy")
    print("=" * 64)
    print(f" 控制台 (浏览器打开) : http://{settings.host}:{settings.port}")
    print("-" * 64)
    print(" 在控制台填入你的 OpenAI 兼容接口与 Key，然后将下方信息填入 Claude Code：")
    print(f"   ANTHROPIC_BASE_URL = {config.anthropic_base_url()}")
    print(f"   ANTHROPIC_API_KEY  = {config.anthropic_api_key}")
    print("=" * 64)

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
