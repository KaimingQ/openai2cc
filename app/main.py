"""FastAPI application exposing the Anthropic Messages API on top of an
OpenAI-compatible backend.

Endpoints:
    POST /v1/messages              -- main chat endpoint (stream + non-stream)
    POST /v1/messages/count_tokens -- rough token counting
    GET  /health                   -- health check
    GET  /                         -- basic info
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import __version__
from .config import settings
from .converter import (
    anthropic_to_openai_request,
    openai_to_anthropic_response,
)
from .models import MessagesRequest, TokenCountRequest
from .runtime_config import config
from .stats import stats
from .streaming import convert_openai_stream_to_anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("proxy")

app = FastAPI(title="OpenAI→Anthropic Proxy", version=__version__)

_STATIC_DIR = Path(__file__).parent / "static"


def _check_auth(x_api_key: str | None, authorization: str | None) -> None:
    """Enforce the generated Anthropic key on the proxied /v1 endpoints."""
    expected = config.anthropic_api_key
    if not expected:
        return
    provided = x_api_key
    if not provided and authorization and authorization.lower().startswith("bearer "):
        provided = authorization[len("bearer "):].strip()
    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _upstream_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.openai_api_key:
        headers["Authorization"] = f"Bearer {config.openai_api_key}"
    return headers


def _content_to_text(content: Any) -> str:
    """Flatten an OpenAI message content (str or parts list) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") == "text":
                    parts.append(p.get("text", ""))
                elif p.get("type") == "image_url":
                    parts.append("[image]")
                else:
                    parts.append(json.dumps(p, ensure_ascii=False))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)


def _extract_input_preview(openai_body: Dict[str, Any]) -> str:
    """Human-readable preview of the newest user turn being sent upstream."""
    messages = openai_body.get("messages") or []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = _content_to_text(msg.get("content"))
            if text.strip():
                return text
    if messages:
        return _content_to_text(messages[-1].get("content"))
    return ""


def _blocks_to_output_preview(content_blocks: Any) -> str:
    """Readable preview of Anthropic response content blocks."""
    parts = []
    for b in content_blocks or []:
        btype = b.get("type")
        if btype == "thinking":
            parts.append("【思考】" + b.get("thinking", ""))
        elif btype == "text":
            parts.append(b.get("text", ""))
        elif btype == "tool_use":
            args = json.dumps(b.get("input", {}), ensure_ascii=False)
            parts.append(f"[tool_use {b.get('name', '')}] {args}")
    return "\n".join(p for p in parts if p)


def _stream_output_preview(output_text: str, thinking_text: str) -> str:
    """Combine streamed reasoning + answer into one preview string."""
    parts = []
    if thinking_text:
        parts.append("【思考】" + thinking_text)
    if output_text:
        parts.append(output_text)
    return "\n".join(parts)


@app.get("/")
async def root() -> Any:
    """Serve the local web UI (falls back to JSON info if UI missing)."""
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return await info()


@app.get("/info")
async def info() -> Dict[str, Any]:
    return {
        "name": "openai-to-anthropic-proxy",
        "version": __version__,
        "upstream": config.base_url,
        "big_model": config.big_model,
        "small_model": config.small_model,
        "ready": config.is_ready(),
        "endpoints": ["/v1/messages", "/v1/messages/count_tokens", "/health"],
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


# --- setup / configuration API --------------------------------------------
@app.get("/config")
async def get_config() -> Dict[str, Any]:
    """Current configuration for the setup page (upstream key masked)."""
    return config.public_dict()


@app.post("/config")
async def save_config(request: Request) -> Dict[str, Any]:
    """Persist edited upstream settings from the setup page."""
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid body: {exc}")
    config.update(body if isinstance(body, dict) else {})
    return config.public_dict()


@app.post("/config/regenerate-key")
async def regenerate_key() -> Dict[str, Any]:
    """Generate a fresh Anthropic API key."""
    config.regenerate_anthropic_key()
    return config.public_dict()


@app.post("/config/test")
async def test_upstream() -> Dict[str, Any]:
    """Send a tiny request upstream to verify the URL + key work."""
    if not config.is_ready():
        return {"ok": False, "message": "请先填写上游 Base URL 与 API Key"}
    url = f"{config.base_url}/chat/completions"
    probe = {
        "model": config.big_model,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, json=probe, headers=_upstream_headers())
    except httpx.RequestError as exc:
        return {"ok": False, "message": f"连接失败: {exc}"}
    if resp.status_code < 400:
        return {"ok": True, "message": f"连接成功 (HTTP {resp.status_code})"}
    return {
        "ok": False,
        "message": f"上游返回 HTTP {resp.status_code}: {resp.text[:200]}",
    }


@app.get("/dashboard/stats")
async def dashboard_stats() -> Dict[str, Any]:
    """Aggregated request statistics for the dashboard."""
    return {
        "config": {
            "upstream": config.base_url,
            "big_model": config.big_model,
            "small_model": config.small_model,
        },
        **stats.snapshot(),
    }


@app.post("/dashboard/reset")
async def dashboard_reset() -> Dict[str, str]:
    """Clear all collected statistics."""
    stats.reset()
    return {"status": "reset"}


@app.post("/v1/messages")
async def create_message(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Any:
    _check_auth(x_api_key, authorization)

    try:
        body = await request.json()
        anthropic_req = MessagesRequest.model_validate(body)
    except Exception as exc:  # noqa: BLE001 - surface parse errors to caller
        logger.warning("Invalid request body: %s", exc)
        raise HTTPException(status_code=400, detail=f"Invalid request: {exc}")

    openai_body = anthropic_to_openai_request(anthropic_req)
    url = f"{config.base_url}/chat/completions"
    mapped_model = openai_body["model"]
    logger.info(
        "%s -> %s (stream=%s)",
        anthropic_req.model,
        mapped_model,
        anthropic_req.stream,
    )
    started = time.perf_counter()

    if anthropic_req.stream:
        return StreamingResponse(
            _stream_upstream(url, openai_body, anthropic_req.model, mapped_model, started),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    return await _complete_upstream(
        url, openai_body, anthropic_req.model, mapped_model, started
    )


async def _complete_upstream(
    url: str,
    openai_body: Dict[str, Any],
    original_model: str,
    mapped_model: str,
    started: float,
) -> JSONResponse:
    """Perform a non-streaming upstream request and translate the response."""
    input_preview = _extract_input_preview(openai_body)
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            resp = await client.post(
                url, json=openai_body, headers=_upstream_headers()
            )
    except httpx.RequestError as exc:
        logger.error("Upstream connection error: %s", exc)
        stats.record(
            requested_model=original_model,
            mapped_model=mapped_model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - started) * 1000,
            stream=False,
            status="error",
            input_preview=input_preview,
            output_preview=f"连接错误: {exc}",
        )
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")

    latency_ms = (time.perf_counter() - started) * 1000
    if resp.status_code >= 400:
        logger.error("Upstream %s: %s", resp.status_code, resp.text[:500])
        stats.record(
            requested_model=original_model,
            mapped_model=mapped_model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=latency_ms,
            stream=False,
            status="error",
            input_preview=input_preview,
            output_preview=f"HTTP {resp.status_code}: {resp.text[:1000]}",
        )
        return JSONResponse(
            status_code=resp.status_code,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": resp.text},
            },
        )

    openai_resp = resp.json()
    anthropic_resp = openai_to_anthropic_response(openai_resp, original_model)
    usage = anthropic_resp.get("usage", {})
    stats.record(
        requested_model=original_model,
        mapped_model=mapped_model,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        latency_ms=latency_ms,
        stream=False,
        status="ok",
        input_preview=input_preview,
        output_preview=_blocks_to_output_preview(anthropic_resp.get("content")),
    )
    return JSONResponse(content=anthropic_resp)


async def _stream_upstream(
    url: str,
    openai_body: Dict[str, Any],
    original_model: str,
    mapped_model: str,
    started: float,
) -> AsyncIterator[str]:
    """Stream from upstream and translate SSE chunks to Anthropic events."""
    message_id = "msg_stream"
    input_preview = _extract_input_preview(openai_body)

    def _on_complete(
        input_tokens: int,
        output_tokens: int,
        finish_reason: Any = None,
        output_text: str = "",
        thinking_text: str = "",
    ) -> None:
        stats.record(
            requested_model=original_model,
            mapped_model=mapped_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
            stream=True,
            status="ok",
            input_preview=input_preview,
            output_preview=_stream_output_preview(output_text, thinking_text),
        )

    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
            async with client.stream(
                "POST", url, json=openai_body, headers=_upstream_headers()
            ) as resp:
                if resp.status_code >= 400:
                    detail = (await resp.aread()).decode("utf-8", "ignore")
                    logger.error("Upstream stream %s: %s", resp.status_code, detail[:500])
                    stats.record(
                        requested_model=original_model,
                        mapped_model=mapped_model,
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=(time.perf_counter() - started) * 1000,
                        stream=True,
                        status="error",
                        input_preview=input_preview,
                        output_preview=f"HTTP {resp.status_code}: {detail[:1000]}",
                    )
                    err = {
                        "type": "error",
                        "error": {"type": "api_error", "message": detail},
                    }
                    yield f"event: error\ndata: {json.dumps(err)}\n\n"
                    return

                async for event in convert_openai_stream_to_anthropic(
                    resp.aiter_lines(), original_model, message_id, _on_complete
                ):
                    yield event
    except httpx.RequestError as exc:
        logger.error("Upstream stream connection error: %s", exc)
        stats.record(
            requested_model=original_model,
            mapped_model=mapped_model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - started) * 1000,
            stream=True,
            status="error",
            input_preview=input_preview,
            output_preview=f"连接错误: {exc}",
        )
        err = {
            "type": "error",
            "error": {"type": "api_error", "message": str(exc)},
        }
        yield f"event: error\ndata: {json.dumps(err)}\n\n"


@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: Request,
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> Dict[str, Any]:
    """Approximate token counting (no upstream call).

    Claude Code calls this to size context. We provide a rough estimate based
    on character length, which is good enough for planning purposes.
    """
    _check_auth(x_api_key, authorization)
    try:
        body = await request.json()
        req = TokenCountRequest.model_validate(body)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid request: {exc}")

    char_count = 0
    if req.system:
        char_count += len(str(req.system))
    for message in req.messages:
        char_count += len(str(message.content))

    # ~4 chars per token is a common heuristic for English/code.
    estimated = max(1, char_count // 4)
    return {"input_tokens": estimated}
