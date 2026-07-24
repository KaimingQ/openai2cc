"""End-to-end smoke test: run a fake OpenAI upstream + the proxy, then send an
Anthropic-style request (non-stream and stream) and check the translation.

Usage:  python tests/e2e_mock.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Point the proxy at our mock upstream before importing it.
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9099/v1"
os.environ["OPENAI_API_KEY"] = "mock-key"
os.environ["PORT"] = "8099"

from app.main import app as proxy_app  # noqa: E402

# --- mock upstream ----------------------------------------------------------
mock = FastAPI()


@mock.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    if body.get("stream"):
        async def gen():
            for piece in ["Hello", ", ", "world"]:
                chunk = {"choices": [{"delta": {"content": piece},
                                      "finish_reason": None}]}
                yield f"data: {json.dumps(chunk)}\n\n"
            final = {"choices": [{"delta": {}, "finish_reason": "stop"}],
                     "usage": {"completion_tokens": 3}}
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return JSONResponse(
        {
            "id": "chatcmpl-1",
            "choices": [
                {"message": {"role": "assistant", "content": "Hello, world"},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3},
        }
    )


def _serve(app, port):
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main() -> None:
    threading.Thread(target=_serve, args=(mock, 9099), daemon=True).start()
    threading.Thread(target=_serve, args=(proxy_app, 8099), daemon=True).start()
    time.sleep(3)

    base = "http://127.0.0.1:8099"
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    }

    # non-streaming
    r = httpx.post(f"{base}/v1/messages", json=payload, timeout=10)
    r.raise_for_status()
    data = r.json()
    assert data["type"] == "message", data
    assert data["content"][0]["text"] == "Hello, world", data
    assert data["stop_reason"] == "end_turn", data
    print("  ok  non-streaming /v1/messages ->", data["content"][0]["text"])

    # streaming
    payload["stream"] = True
    with httpx.stream("POST", f"{base}/v1/messages", json=payload, timeout=10) as s:
        events = "".join(chunk for chunk in s.iter_text())
    assert "message_start" in events
    assert "text_delta" in events
    assert "message_stop" in events
    assert "Hello" in events and "world" in events
    print("  ok  streaming /v1/messages -> received full SSE sequence")

    # count_tokens
    r = httpx.post(f"{base}/v1/messages/count_tokens", json=payload, timeout=10)
    r.raise_for_status()
    assert "input_tokens" in r.json()
    print("  ok  /v1/messages/count_tokens ->", r.json())

    # web UI is served at root
    r = httpx.get(f"{base}/", timeout=10)
    r.raise_for_status()
    assert "OpenAI → Anthropic" in r.text
    print("  ok  GET / -> web UI served")

    # dashboard stats reflect the requests we just made
    r = httpx.get(f"{base}/dashboard/stats", timeout=10)
    r.raise_for_status()
    stats = r.json()
    assert stats["totals"]["total_requests"] >= 2, stats
    assert stats["totals"]["total_output_tokens"] >= 3, stats
    print("  ok  /dashboard/stats ->", stats["totals"])

    print("\nE2E mock test passed.")


if __name__ == "__main__":
    main()
