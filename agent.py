"""Top-level entry point exposing the required interface.

    from agent import Agent
    a = Agent()
    a.next("Hi")  # -> {"message": "..."}

The implementation lives in the `payment_agent` package under `src/`; this
module simply re-exports the `Agent` class so the evaluator can import it
directly from the repository root as specified in the assignment.
"""

from __future__ import annotations

import os
import sys

# Make the packaged implementation importable when running from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from payment_agent import Agent  # noqa: E402

__all__ = ["Agent"]
