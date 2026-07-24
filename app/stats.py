"""In-memory + on-disk request statistics for the proxy dashboard.

Every proxied request records: timestamp, requested (Anthropic) model, mapped
(OpenAI) model, input/output tokens, latency and status. Aggregates are kept
in memory and periodically persisted to a JSON file so they survive restarts.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List

_RECENT_LIMIT = 200
_PREVIEW_LIMIT = 4000  # max chars stored per input/output content preview


def _clip(text: Any, limit: int = _PREVIEW_LIMIT) -> str:
    """Coerce to string and clip to ``limit`` chars, marking truncation."""
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n…（已截断，共 {len(s)} 字）"


class StatsStore:
    """Thread-safe aggregate + recent-log store for proxied requests."""

    def __init__(self, path: str = "stats.json") -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._totals: Dict[str, Any] = {
            "total_requests": 0,
            "total_errors": 0,
            "total_stream": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_latency_ms": 0.0,
        }
        self._by_model: Dict[str, Dict[str, Any]] = {}
        self._recent: Deque[Dict[str, Any]] = deque(maxlen=_RECENT_LIMIT)
        self._load()

    # --- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self._totals.update(data.get("totals", {}))
        self._by_model = data.get("by_model", {})
        for item in data.get("recent", []):
            self._recent.append(item)

    def _save_locked(self) -> None:
        payload = {
            "totals": self._totals,
            "by_model": self._by_model,
            "recent": list(self._recent),
        }
        try:
            self._path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), "utf-8"
            )
        except OSError:
            pass

    # --- recording ---------------------------------------------------------
    def record(
        self,
        *,
        requested_model: str,
        mapped_model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        stream: bool,
        status: str,
        input_preview: str = "",
        output_preview: str = "",
    ) -> None:
        """Record one completed request."""
        with self._lock:
            self._totals["total_requests"] += 1
            self._totals["total_input_tokens"] += int(input_tokens or 0)
            self._totals["total_output_tokens"] += int(output_tokens or 0)
            self._totals["total_latency_ms"] += float(latency_ms or 0)
            if stream:
                self._totals["total_stream"] += 1
            if status != "ok":
                self._totals["total_errors"] += 1

            bucket = self._by_model.setdefault(
                requested_model,
                {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "errors": 0,
                    "mapped_model": mapped_model,
                },
            )
            bucket["requests"] += 1
            bucket["input_tokens"] += int(input_tokens or 0)
            bucket["output_tokens"] += int(output_tokens or 0)
            bucket["mapped_model"] = mapped_model
            if status != "ok":
                bucket["errors"] += 1

            self._recent.appendleft(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "requested_model": requested_model,
                    "mapped_model": mapped_model,
                    "input_tokens": int(input_tokens or 0),
                    "output_tokens": int(output_tokens or 0),
                    "latency_ms": round(float(latency_ms or 0), 1),
                    "stream": stream,
                    "status": status,
                    "input_preview": _clip(input_preview),
                    "output_preview": _clip(output_preview),
                }
            )
            self._save_locked()

    # --- reads -------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            total = self._totals["total_requests"] or 1
            avg_latency = self._totals["total_latency_ms"] / total
            return {
                "totals": {
                    **self._totals,
                    "avg_latency_ms": round(avg_latency, 1),
                },
                "by_model": self._by_model,
                "recent": list(self._recent),
            }

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._recent)[:limit]

    def reset(self) -> None:
        with self._lock:
            for key in self._totals:
                self._totals[key] = 0 if key != "total_latency_ms" else 0.0
            self._by_model.clear()
            self._recent.clear()
            self._save_locked()


stats = StatsStore()
