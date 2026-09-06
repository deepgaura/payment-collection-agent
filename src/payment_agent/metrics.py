"""
WHAT THIS FILE IS (in one line):
    A tiny notebook that records numbers about the conversation, so we can see
    how the agent is doing.

IT TRACKS TWO KINDS OF THINGS:
    1. COUNTERS  = "how many times did X happen?"
                   e.g. verification.success = 1, payment.failure = 2
    2. TIMERS    = "how long did X take (in milliseconds)?"
                   e.g. the LLM call took 2000 ms, the whole turn took 3000 ms

WHY IT EXISTS:
    So at the end you can print a little report like:
        latency (ms):
          turn        count=5  avg=3000  max=8000
          llm         count=5  avg=2000  max=3200
        counters:
          verification.success   1
          payment.success        1
    This is "observability" - knowing what your system is doing. It's kept
    super small on purpose (no external tools, all in memory).
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class _Timings:
    """
    Stats for ONE timer (e.g. "how long the LLM takes").

    Instead of storing every single measurement, we just keep a running tally:
      - count    : how many times we measured
      - total_ms : all the times added up
      - max_ms   : the slowest single time seen
    From these we can compute the average cheaply.
    """

    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0

    def add(self, ms: float) -> None:
        # Record one new measurement (in milliseconds).
        # Example: add(2000) means "that took 2 seconds".
        self.count += 1
        self.total_ms += ms
        self.max_ms = max(self.max_ms, ms)

    @property
    def avg_ms(self) -> float:
        # Average time = total / count. (Guard against divide-by-zero.)
        return self.total_ms / self.count if self.count else 0.0


@dataclass
class Metrics:
    """The notebook itself: all counters and all timers for one conversation."""

    # `counters`: name -> number.   e.g. {"turns": 5, "payment.success": 1}
    # `timings` : name -> _Timings. e.g. {"turn": <stats>, "llm": <stats>}
    # defaultdict means: if you use a name that doesn't exist yet, it's created
    # automatically starting at 0 (so we never have to check "does it exist?").
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    timings: dict[str, _Timings] = field(default_factory=lambda: defaultdict(_Timings))

    # --- writing numbers in ------------------------------------------------ #
    def incr(self, name: str, by: int = 1) -> None:
        # Add 1 (or `by`) to a counter.
        # Example: incr("payment.success") -> payment.success goes 0 -> 1.
        self.counters[name] += by

    def record_ms(self, name: str, ms: float) -> None:
        # Record one timing measurement under a name.
        # Example: record_ms("llm", 2000) -> logs a 2-second LLM call.
        self.timings[name].add(ms)

    @contextmanager
    def timer(self, name: str):
        """
        A convenient stopwatch you wrap around a block of code, like:

            with metrics.timer("turn"):
                do_the_work()      # <- whatever runs here gets timed

        It notes the start time, lets the code run, then (even if it errors)
        records how long it took. No manual start/stop needed.
        """
        start = time.perf_counter()          # start the stopwatch
        try:
            yield                            # <- the wrapped code runs here
        finally:
            # stop the stopwatch and save the elapsed time (converted to ms).
            self.record_ms(name, (time.perf_counter() - start) * 1000.0)

    # --- reading numbers out ----------------------------------------------- #
    def snapshot(self) -> dict:
        """
        Return everything as a plain dictionary (easy to turn into JSON or send
        somewhere). Example output:
            {
              "counters": {"turns": 5, "payment.success": 1},
              "timings_ms": {"turn": {"count": 5, "avg": 3000.0, "max": 8000.0}}
            }
        """
        return {
            "counters": dict(self.counters),
            "timings_ms": {
                k: {"count": t.count, "avg": round(t.avg_ms, 1), "max": round(t.max_ms, 1)}
                for k, t in self.timings.items()
            },
        }

    def render(self) -> str:
        """
        Build a nice human-readable text report (what the CLI prints at the end).
        Just loops over the timers and counters and formats them into lines.
        """
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
