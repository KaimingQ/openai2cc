"""Unit tests for the Anthropic <-> OpenAI translation layer.

Run with:  python -m pytest -q   (or)   python tests/test_conversion.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use an isolated (non-existent) config path so tests are deterministic and
# never read/write a real config.json.
os.environ["CONFIG_PATH"] = os.path.join(tempfile.gettempdir(), "o2a_test_config.json")
if os.path.exists(os.environ["CONFIG_PATH"]):
    os.remove(os.environ["CONFIG_PATH"])

from app.converter import (  # noqa: E402
    anthropic_to_openai_request,
    map_finish_reason,
    openai_to_anthropic_response,
)
from app.models import MessagesRequest  # noqa: E402
from app.streaming import convert_openai_stream_to_anthropic  # noqa: E402


def test_simple_text_request() -> None:
    req = MessagesRequest.model_validate(
        {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "system": "You are helpful",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )
    body = anthropic_to_openai_request(req)
    assert body["messages"][0] == {"role": "system", "content": "You are helpful"}
    assert body["messages"][1] == {"role": "user", "content": "Hello"}
    assert body["max_tokens"] == 100


def test_tool_use_roundtrip() -> None:
    req = MessagesRequest.model_validate(
        {
            "model": "claude-3-5-haiku-20241022",
            "max_tokens": 50,
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "SF"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "sunny",
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object"},
                }
            ],
        }
    )
    body = anthropic_to_openai_request(req)
    # haiku maps to the small model
    assert body["model"]  # mapped, not empty
    assistant = body["messages"][1]
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    tool_msg = body["messages"][2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "toolu_1"
    assert tool_msg["content"] == "sunny"
    assert body["tools"][0]["function"]["name"] == "get_weather"


def test_response_conversion_text() -> None:
    openai_resp = {
        "id": "chatcmpl-abc",
        "choices": [
            {"message": {"role": "assistant", "content": "Hi there"},
             "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    result = openai_to_anthropic_response(openai_resp, "claude-3-5-sonnet")
    assert result["type"] == "message"
    assert result["content"][0] == {"type": "text", "text": "Hi there"}
    assert result["stop_reason"] == "end_turn"
    assert result["usage"] == {"input_tokens": 10, "output_tokens": 3}
    assert result["id"].startswith("msg_")


def test_response_conversion_tool_call() -> None:
    openai_resp = {
        "id": "x",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "SF"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 8},
    }
    result = openai_to_anthropic_response(openai_resp, "claude-3-5-sonnet")
    block = result["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "get_weather"
    assert block["input"] == {"city": "SF"}
    assert result["stop_reason"] == "tool_use"


def test_response_conversion_reasoning() -> None:
    # Reasoning models (e.g. DeepSeek) return chain-of-thought in
    # ``reasoning_content``; it should become an Anthropic thinking block
    # placed before the answer text.
    openai_resp = {
        "id": "chatcmpl-r",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "reasoning_content": "Let me think...",
                    "content": "The answer is 2",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 9},
    }
    result = openai_to_anthropic_response(openai_resp, "claude-3-5-sonnet")
    assert result["content"][0] == {"type": "thinking", "thinking": "Let me think..."}
    assert result["content"][1] == {"type": "text", "text": "The answer is 2"}
    assert result["stop_reason"] == "end_turn"


def test_finish_reason_map() -> None:
    assert map_finish_reason("stop") == "end_turn"
    assert map_finish_reason("length") == "max_tokens"
    assert map_finish_reason("tool_calls") == "tool_use"
    assert map_finish_reason(None) == "end_turn"


def test_streaming_text() -> None:
    async def fake_lines():
        chunks = [
            {"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"completion_tokens": 2}},
        ]
        for c in chunks:
            yield f"data: {json.dumps(c)}"
        yield "data: [DONE]"

    async def run():
        events = []
        async for e in convert_openai_stream_to_anthropic(
            fake_lines(), "claude-3-5-sonnet", "msg_test"
        ):
            events.append(e)
        return events

    events = asyncio.run(run())
    joined = "".join(events)
    assert "message_start" in joined
    assert "content_block_start" in joined
    assert "text_delta" in joined
    assert "Hel" in joined and "lo" in joined
    assert "message_stop" in joined
    assert '"stop_reason": "end_turn"' in joined
    assert '"output_tokens": 2' in joined


def test_streaming_reasoning() -> None:
    # A reasoning stream: reasoning_content deltas first, then content deltas.
    async def fake_lines():
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "think"}, "finish_reason": None}]},
            {"choices": [{"delta": {"reasoning_content": "ing"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "answer"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"completion_tokens": 3}},
        ]
        for c in chunks:
            yield f"data: {json.dumps(c)}"
        yield "data: [DONE]"

    async def run():
        events = []
        async for e in convert_openai_stream_to_anthropic(
            fake_lines(), "claude-3-5-sonnet", "msg_test"
        ):
            events.append(e)
        return events

    joined = "".join(asyncio.run(run()))
    # thinking block (index 0) opened before the text block (index 1)
    assert '"content_block": {"type": "thinking"' in joined
    assert '"type": "thinking_delta", "thinking": "think"' in joined
    assert '"content_block": {"type": "text"' in joined
    assert "answer" in joined
    # thinking block starts at index 0, text at index 1
    assert '"content_block_start", "index": 0' in joined
    assert '"content_block_start", "index": 1' in joined


def _run_all() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\nAll {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
