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
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from .converter import THINK_CLOSE, THINK_OPEN, map_finish_reason


def _sse(event: str, data: Dict[str, Any]) -> str:
    """Format a single Server-Sent Event line pair."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _hold_back(buf: str, tag: str) -> str:
    """Return the prefix of ``buf`` that is safe to emit now.

    Any trailing substring of ``buf`` that could be the start of ``tag`` is
    withheld so a tag split across stream chunks is never emitted verbatim.
    """
    for k in range(min(len(buf), len(tag) - 1), 0, -1):
        if buf.endswith(tag[:k]):
            return buf[: len(buf) - k]
    return buf


class _StreamState:
    """Mutable bookkeeping for an in-flight stream translation."""

    def __init__(self) -> None:
        self.next_index = 0
        self.thinking_index: Optional[int] = None
        self.thinking_started = False
        self.thinking_closed = False
        self.text_index: Optional[int] = None
        self.text_started = False
        # OpenAI tool_call index -> Anthropic content block index
        self.tool_index_map: Dict[int, int] = {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.finish_reason: Optional[str] = None
        # accumulated visible text + reasoning for statistics / detail view
        self.text_pieces: list[str] = []
        self.thinking_pieces: list[str] = []
        # inline <think>...</think> splitter state (e.g. MiniMax)
        self.tag_buf = ""
        self.in_think = False


def _split_stream_text(state: "_StreamState", piece: str) -> List[Tuple[str, str]]:
    """Feed a ``content`` text piece through the inline-think splitter.

    Returns a list of ``(kind, text)`` segments where ``kind`` is ``"thinking"``
    or ``"text"``. ``<think>`` / ``</think>`` tags are consumed (never emitted),
    even when split across chunks.
    """
    state.tag_buf += piece
    out: List[Tuple[str, str]] = []
    while state.tag_buf:
        if state.in_think:
            idx = state.tag_buf.find(THINK_CLOSE)
            if idx == -1:
                safe = _hold_back(state.tag_buf, THINK_CLOSE)
                if safe:
                    out.append(("thinking", safe))
                    state.tag_buf = state.tag_buf[len(safe):]
                break
            if idx > 0:
                out.append(("thinking", state.tag_buf[:idx]))
            state.tag_buf = state.tag_buf[idx + len(THINK_CLOSE):]
            state.in_think = False
        else:
            idx = state.tag_buf.find(THINK_OPEN)
            if idx == -1:
                safe = _hold_back(state.tag_buf, THINK_OPEN)
                if safe:
                    out.append(("text", safe))
                    state.tag_buf = state.tag_buf[len(safe):]
                break
            if idx > 0:
                out.append(("text", state.tag_buf[:idx]))
            state.tag_buf = state.tag_buf[idx + len(THINK_OPEN):]
            state.in_think = True
    return out


def _flush_stream_text(state: "_StreamState") -> List[Tuple[str, str]]:
    """Emit whatever remains buffered when the stream ends."""
    if not state.tag_buf:
        return []
    kind = "thinking" if state.in_think else "text"
    out = [(kind, state.tag_buf)]
    state.tag_buf = ""
    return out


def _emit_thinking(state: "_StreamState", text: str) -> List[str]:
    """Emit a reasoning segment as an Anthropic ``thinking`` block delta."""
    # Anthropic blocks can't interleave: once text has started, any late
    # reasoning is degraded to visible text to stay protocol-compliant.
    if state.text_started:
        return _emit_text(state, text)
    out: List[str] = []
    if not state.thinking_started:
        state.thinking_index = state.next_index
        state.next_index += 1
        state.thinking_started = True
        out.append(
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": state.thinking_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            )
        )
    out.append(
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": state.thinking_index,
                "delta": {"type": "thinking_delta", "thinking": text},
            },
        )
    )
    state.thinking_pieces.append(text)
    return out


def _emit_text(state: "_StreamState", text: str) -> List[str]:
    """Emit a visible segment as an Anthropic ``text`` block delta."""
    out: List[str] = []
    if state.thinking_started and not state.thinking_closed:
        state.thinking_closed = True
        out.append(
            _sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": state.thinking_index},
            )
        )
    if not state.text_started:
        state.text_index = state.next_index
        state.next_index += 1
        state.text_started = True
        out.append(
            _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": state.text_index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        )
    out.append(
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": state.text_index,
                "delta": {"type": "text_delta", "text": text},
            },
        )
    )
    state.text_pieces.append(text)
    return out


async def convert_openai_stream_to_anthropic(
    openai_lines: AsyncIterator[str],
    original_model: str,
    message_id: str,
    on_complete: Optional[Callable[..., None]] = None,
) -> AsyncIterator[str]:
    """Yield Anthropic SSE strings translated from an OpenAI SSE line stream.

    ``on_complete`` (if given) is called once at the end with keyword args
    ``(input_tokens, output_tokens, finish_reason, output_text, thinking_text)``
    for statistics / the request detail view.
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

        # --- reasoning (thinking) delta ---
        # Some OpenAI-compatible providers (e.g. DeepSeek reasoning models)
        # stream chain-of-thought separately in ``reasoning_content``; surface
        # it as an Anthropic ``thinking`` block.
        reasoning_piece = delta.get("reasoning_content")
        if isinstance(reasoning_piece, str) and reasoning_piece:
            for event in _emit_thinking(state, reasoning_piece):
                yield event

        # --- text delta ---
        # ``content`` may carry inline <think>...</think> reasoning (e.g.
        # MiniMax); split it out so tags never leak into the visible answer.
        text_piece = delta.get("content")
        if isinstance(text_piece, str) and text_piece:
            for kind, seg in _split_stream_text(state, text_piece):
                if not seg:
                    continue
                emit = _emit_thinking if kind == "thinking" else _emit_text
                for event in emit(state, seg):
                    yield event

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

    # 2. flush any buffered tail, then close open content blocks
    for kind, seg in _flush_stream_text(state):
        if not seg:
            continue
        emit = _emit_thinking if kind == "thinking" else _emit_text
        for event in emit(state, seg):
            yield event
    if state.thinking_started and not state.thinking_closed:
        state.thinking_closed = True
        yield _sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": state.thinking_index},
        )
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
        on_complete(
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            finish_reason=state.finish_reason,
            output_text="".join(state.text_pieces),
            thinking_text="".join(state.thinking_pieces),
        )


async def error_stream(message: str) -> AsyncIterator[str]:
    """Emit a minimal Anthropic-style error event stream."""
    yield _sse(
        "error",
        {"type": "error", "error": {"type": "api_error", "message": message}},
    )
