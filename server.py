"""Convenience entry point: ``python server.py`` starts the proxy."""
from __future__ import annotations

import uvicorn

from app.config import settings


def main() -> None:
    print("=" * 60)
    print(" OpenAI → Anthropic Proxy")
    print("=" * 60)
    print(f" Listening : http://{settings.host}:{settings.port}")
    print(f" Upstream  : {settings.base_url}")
    print(f" Big model : {settings.big_model}")
    print(f" Small     : {settings.small_model}")
    print("-" * 60)
    print(" Point Claude Code at this server, e.g.:")
    print(f"   export ANTHROPIC_BASE_URL=http://{settings.host}:{settings.port}")
    print("   export ANTHROPIC_API_KEY=any-value")
    print("   claude")
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
