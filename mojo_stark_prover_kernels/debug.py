"""Debug tracing primitives with bounded memory usage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

MAX_DEBUG_EVENT_CHARS = 512


@dataclass
class DebugTracer:
    level: int = 0
    max_events: int = 10000
    _events: list[str] = field(default_factory=list)
    _saturated: bool = False

    def emit(self, level: int, message: str) -> None:
        if self.level < level:
            return
        if self._saturated:
            return
        if len(self._events) >= self.max_events:
            self._events.append("WARN trace saturated; dropping additional events")
            self._saturated = True
            return
        ts = datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds")
        msg = message.strip().replace("\n", "\\n")
        if len(msg) > MAX_DEBUG_EVENT_CHARS:
            msg = msg[:MAX_DEBUG_EVENT_CHARS] + "...<truncated>"
        self._events.append(f"{ts} L{level} {msg}")

    def snapshot(self) -> tuple[str, ...]:
        return tuple(self._events)

