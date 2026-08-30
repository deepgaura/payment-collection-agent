"""Lightweight, dependency-free metrics collection.

This is NOT a full observability stack (no Prometheus/dashboards) - it is an
in-memory collector that records the signals worth watching for an agent like
this, so they can be printed at the end of a session or scraped/forwarded in a
real deployment. Kept deliberately small: one counter/timer store, thread-unsafe
(a conversation is single-threaded), no external dependencies.

What we track and why:
- turn latency        : end-to-end responsiveness the user feels.
- llm latency + calls : the dominant cost/latency source; also fallback rate.
- tool latency + calls: per-endpoint API latency and error-code breakdown.
- counters            : verification outcomes, retries, payment outcomes.

Usage:
    m = Metrics()
    with m.timer("turn"):
        ...
    m.incr("verification.success")
    print(m.render())
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class _Timings:
    """Running stats for a set of durations (ms), without storing every sample."""

    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def add(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        self.max_ms = max(self.max_ms, ms)

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count else 0.0


@dataclass
class Metrics:
    """In-memory counters and timers for one agent/session."""

    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    timings: dict[str, _Timings] = field(default_factory=lambda: defaultdict(_Timings))

    # --- recording ---
    def incr(self, name: str, by: int = 1) -> None:
        self.counters[name] += by

    def record_ms(self, name: str, ms: float) -> None:
        self.timings[name].add(ms)

    @contextmanager
    def timer(self, name: str):
        """Time a block of code and record its duration in milliseconds."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record_ms(name, (time.perf_counter() - start) * 1000.0)

    # --- reading ---
    def snapshot(self) -> dict:
        """A plain-dict view, suitable for logging/JSON/forwarding."""
        return {
            "counters": dict(self.counters),
            "timings_ms": {
                k: {"count": t.count, "avg": round(t.avg_ms, 1), "max": round(t.max_ms, 1)}
                for k, t in self.timings.items()
            },
        }

    def render(self) -> str:
        """Human-readable summary (for the CLI / debugging)."""
        lines = ["--- metrics ---"]
        if self.timings:
            lines.append("latency (ms):")
            for k in sorted(self.timings):
                t = self.timings[k]
                lines.append(f"  {k:<16} count={t.count:<4} avg={t.avg_ms:6.1f} max={t.max_ms:6.1f}")
        if self.counters:
            lines.append("counters:")
            for k in sorted(self.counters):
                lines.append(f"  {k:<28} {self.counters[k]}")
        return "\n".join(lines)
