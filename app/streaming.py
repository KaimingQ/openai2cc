"""Streaming translation: OpenAI SSE chunks -> Anthropic SSE events.

Claude Code expects the Anthropic event sequence:

    message_start
    content_block_start / content_block_delta / content_block_stop  (repeated)
    message_delta (stop_reason + usage)
    message_stop

We consume the OpenAI ``chat.completions`` stream (delta chunks) and emit the
corresponding Anthropic events, tracking one text block plus any number of
tool_use blocks.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Callable, Dict, Optional

from .converter import map_finish_reason


def _sse(event: str, data: Dict[str, Any]) -> str:
    """Format a single Server-Sent Event line pair."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class _StreamState:
    """Mutable bookkeeping for an in-flight stream translation."""

    def __init__(self) -> None:
        self.next_index = 0
        self.text_index: Optional[int] = None
        self.text_started = False
        # OpenAI tool_call index -> Anthropic content block index
        self.tool_index_map: Dict[int, int] = {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.finish_reason: Optional[str] = None


async def convert_openai_stream_to_anthropic(
    openai_lines: AsyncIterator[str],
    original_model: str,
    message_id: str,
    on_complete: Optional[Callable[[int, int, Optional[str]], None]] = None,
) -> AsyncIterator[str]:
    """Yield Anthropic SSE strings translated from an OpenAI SSE line stream.

    ``on_complete`` (if given) is called once at the end with
    ``(input_tokens, output_tokens, finish_reason)`` for statistics.
    """
    state = _StreamState()

    # 1. message_start
    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": original_model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    yield _sse("ping", {"type": "ping"})

    async for raw_line in openai_lines:
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        # Usage may arrive on a chunk with empty choices (stream_options).
        usage = chunk.get("usage")
        if usage:
            if usage.get("completion_tokens") is not None:
                state.output_tokens = usage.get("completion_tokens", state.output_tokens)
            if usage.get("prompt_tokens") is not None:
                state.input_tokens = usage.get("prompt_tokens", state.input_tokens)

        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta", {}) or {}

        # --- text delta ---
        text_piece = delta.get("content")
        if isinstance(text_piece, str) and text_piece:
            if not state.text_started:
                state.text_index = state.next_index
                state.next_index += 1
                state.text_started = True
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": state.text_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": state.text_index,
                    "delta": {"type": "text_delta", "text": text_piece},
                },
            )

        # --- tool call deltas ---
        for tool_call in delta.get("tool_calls") or []:
            oai_index = tool_call.get("index", 0)
            fn = tool_call.get("function", {}) or {}

            if oai_index not in state.tool_index_map:
                block_index = state.next_index
                state.next_index += 1
                state.tool_index_map[oai_index] = block_index
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_call.get("id") or f"toolu_{block_index}",
                            "name": fn.get("name", ""),
                            "input": {},
                        },
                    },
                )

            block_index = state.tool_index_map[oai_index]
            args_piece = fn.get("arguments")
            if args_piece:
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": args_piece,
                        },
                    },
                )

        if choice.get("finish_reason"):
            state.finish_reason = choice["finish_reason"]

    # 2. close any open content blocks
    if state.text_started and state.text_index is not None:
        yield _sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": state.text_index},
        )
    for block_index in state.tool_index_map.values():
        yield _sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": block_index},
        )

    # 3. message_delta with stop reason + usage
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": map_finish_reason(state.finish_reason),
                "stop_sequence": None,
            },
            "usage": {"output_tokens": state.output_tokens},
        },
    )

    # 4. message_stop
    yield _sse("message_stop", {"type": "message_stop"})

    if on_complete is not None:
        on_complete(state.input_tokens, state.output_tokens, state.finish_reason)


async def error_stream(message: str) -> AsyncIterator[str]:
    """Emit a minimal Anthropic-style error event stream."""
    yield _sse(
        "error",
        {"type": "error", "error": {"type": "api_error", "message": message}},
    )
