"""Translation between the Anthropic Messages API and the OpenAI Chat
Completions API.

Two directions are handled:

* ``anthropic_to_openai_request`` -- convert an incoming Anthropic request
  (what Claude Code sends) into an OpenAI-compatible chat completion request.
* ``openai_to_anthropic_response`` -- convert a non-streaming OpenAI response
  back into the Anthropic response shape.

Streaming translation lives in :mod:`app.streaming`.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .config import settings
from .models import MessagesRequest
from .runtime_config import config

# --- finish reason mapping (OpenAI -> Anthropic) ---------------------------
_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def map_finish_reason(reason: Optional[str]) -> str:
    """Map an OpenAI finish_reason to an Anthropic stop_reason."""
    if reason is None:
        return "end_turn"
    return _FINISH_REASON_MAP.get(reason, "end_turn")


# --- helpers ----------------------------------------------------------------
def _system_to_text(system: Any) -> Optional[str]:
    """Flatten an Anthropic system prompt (str or block list) to plain text."""
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    if isinstance(system, list):
        parts: List[str] = []
        for block in system:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n\n".join(p for p in parts if p) or None
    return None


def _block_get(block: Any, key: str, default: Any = None) -> Any:
    """Access a field from a content block whether it is a dict or model."""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _stringify_tool_result(content: Any) -> str:
    """Anthropic tool_result content -> a string OpenAI's tool role expects."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _image_block_to_openai(block: Any) -> Optional[Dict[str, Any]]:
    """Convert an Anthropic image block to an OpenAI image_url part."""
    source = _block_get(block, "source")
    if not source:
        return None
    src = source if isinstance(source, dict) else source.__dict__
    if src.get("type") == "base64":
        media_type = src.get("media_type", "image/png")
        data = src.get("data", "")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{data}"},
        }
    if src.get("type") == "url":
        return {"type": "image_url", "image_url": {"url": src.get("url", "")}}
    return None


# --- request conversion -----------------------------------------------------
def _convert_user_message(content: Any) -> List[Dict[str, Any]]:
    """Convert one Anthropic user message into one or more OpenAI messages.

    A single Anthropic user turn may carry tool_result blocks which OpenAI
    represents as separate ``role: tool`` messages, so we may emit several.
    """
    if isinstance(content, str):
        return [{"role": "user", "content": content}]

    text_parts: List[Dict[str, Any]] = []
    tool_messages: List[Dict[str, Any]] = []

    for block in content:
        btype = _block_get(block, "type")
        if btype == "text":
            text_parts.append({"type": "text", "text": _block_get(block, "text", "")})
        elif btype == "image":
            img = _image_block_to_openai(block)
            if img:
                text_parts.append(img)
        elif btype == "tool_result":
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": _block_get(block, "tool_use_id", ""),
                    "content": _stringify_tool_result(_block_get(block, "content")),
                }
            )

    messages: List[Dict[str, Any]] = []
    if text_parts:
        # Collapse a lone text part into a plain string for wider compatibility.
        if len(text_parts) == 1 and text_parts[0].get("type") == "text":
            messages.append({"role": "user", "content": text_parts[0]["text"]})
        else:
            messages.append({"role": "user", "content": text_parts})
    messages.extend(tool_messages)
    return messages


def _convert_assistant_message(content: Any) -> Dict[str, Any]:
    """Convert one Anthropic assistant message into an OpenAI assistant msg."""
    if isinstance(content, str):
        return {"role": "assistant", "content": content}

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in content:
        btype = _block_get(block, "type")
        if btype == "text":
            text_parts.append(_block_get(block, "text", ""))
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": _block_get(block, "id", ""),
                    "type": "function",
                    "function": {
                        "name": _block_get(block, "name", ""),
                        "arguments": json.dumps(
                            _block_get(block, "input", {}) or {},
                            ensure_ascii=False,
                        ),
                    },
                }
            )

    msg: Dict[str, Any] = {"role": "assistant"}
    text = "".join(text_parts)
    # OpenAI requires content to be present; null is allowed when tool_calls set.
    msg["content"] = text if text else None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _convert_tools(tools: Optional[List[Any]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        name = _block_get(tool, "name")
        if not name:
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": _block_get(tool, "description") or "",
                    "parameters": _block_get(tool, "input_schema") or {"type": "object"},
                },
            }
        )
    return converted or None


def _convert_tool_choice(tool_choice: Optional[Dict[str, Any]]) -> Optional[Any]:
    if not tool_choice:
        return None
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    if choice_type == "none":
        return "none"
    return "auto"


def anthropic_to_openai_request(req: MessagesRequest) -> Dict[str, Any]:
    """Build an OpenAI chat-completion request body from an Anthropic request."""
    openai_messages: List[Dict[str, Any]] = []

    system_text = _system_to_text(req.system)
    if system_text:
        openai_messages.append({"role": "system", "content": system_text})

    for message in req.messages:
        if message.role == "user":
            openai_messages.extend(_convert_user_message(message.content))
        elif message.role == "assistant":
            openai_messages.append(_convert_assistant_message(message.content))

    max_tokens = req.max_tokens
    if settings.max_tokens_limit > 0:
        max_tokens = min(max_tokens, settings.max_tokens_limit)

    body: Dict[str, Any] = {
        "model": config.map_model(req.model),
        "messages": openai_messages,
        "max_tokens": max_tokens,
        "stream": req.stream,
    }

    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stop_sequences:
        body["stop"] = req.stop_sequences

    tools = _convert_tools(req.tools)
    if tools:
        body["tools"] = tools
        tool_choice = _convert_tool_choice(req.tool_choice)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

    if req.stream:
        # Ask upstream to include token usage in the final stream chunk.
        body["stream_options"] = {"include_usage": True}

    return body


# --- response conversion (non-streaming) -----------------------------------
def openai_to_anthropic_response(
    openai_resp: Dict[str, Any], original_model: str
) -> Dict[str, Any]:
    """Convert a non-streaming OpenAI response into the Anthropic shape."""
    choices = openai_resp.get("choices") or [{}]
    choice = choices[0]
    message = choice.get("message", {}) or {}

    content_blocks: List[Dict[str, Any]] = []

    # Reasoning models (e.g. DeepSeek) expose chain-of-thought separately;
    # surface it as an Anthropic ``thinking`` block ahead of the answer.
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        content_blocks.append({"type": "thinking", "thinking": reasoning})

    text = message.get("content")
    if isinstance(text, str) and text:
        content_blocks.append({"type": "text", "text": text})
    elif isinstance(text, list):
        # Some providers return content parts; extract text.
        for part in text:
            if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                content_blocks.append(
                    {"type": "text", "text": part.get("text", "")}
                )

    for tool_call in message.get("tool_calls") or []:
        fn = tool_call.get("function", {}) or {}
        raw_args = fn.get("arguments") or "{}"
        try:
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, TypeError):
            parsed_args = {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or f"toolu_{len(content_blocks)}",
                "name": fn.get("name", ""),
                "input": parsed_args,
            }
        )

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    usage = openai_resp.get("usage", {}) or {}
    message_id = openai_resp.get("id") or "msg_unknown"
    if not str(message_id).startswith("msg_"):
        message_id = f"msg_{message_id}"

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": content_blocks,
        "stop_reason": map_finish_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }
