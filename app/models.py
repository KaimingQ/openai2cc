"""Pydantic models describing the Anthropic Messages API request shape.

Only the fields relevant to translation are modelled; unknown fields are
ignored so we stay forward-compatible with Claude Code payloads.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ContentBlockText(BaseModel):
    type: str = "text"
    text: str


class ContentBlockImageSource(BaseModel):
    type: str  # "base64"
    media_type: str
    data: str


class ContentBlockImage(BaseModel):
    type: str = "image"
    source: ContentBlockImageSource


class ContentBlockToolUse(BaseModel):
    type: str = "tool_use"
    id: str
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)


class ContentBlockToolResult(BaseModel):
    type: str = "tool_result"
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], None] = None
    is_error: Optional[bool] = None


# A content block can be any of the above; we keep it loose as dicts too.
ContentBlock = Union[
    ContentBlockText,
    ContentBlockImage,
    ContentBlockToolUse,
    ContentBlockToolResult,
    Dict[str, Any],
]


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: Union[str, List[Any]]


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)


class ThinkingConfig(BaseModel):
    type: Optional[str] = None
    budget_tokens: Optional[int] = None


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[Any]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None


class TokenCountRequest(BaseModel):
    """Payload for the /v1/messages/count_tokens endpoint."""

    model: str
    messages: List[Message]
    system: Optional[Union[str, List[Any]]] = None
    tools: Optional[List[Tool]] = None
