"""Interactive REPL for the payment agent.

Usage:
    python cli.py

Type your messages at the prompt. Type Ctrl-C or 'quit' to exit. This is a
convenience wrapper for manual testing; the evaluator uses `Agent.next()`
directly.
"""

from __future__ import annotations

import logging

from agent import Agent
from payment_agent.state import Step


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    agent = Agent()
    print("Payment Collection Agent (type 'quit' to exit)\n")
    # Kick off with a greeting turn.
    first = agent.next("")
    print(f"agent> {first['message']}")

    while True:
        try:
            user = input("you  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nagent> Session ended. Take care!")
            break
        response = agent.next(user)
        print(f"agent> {response['message']}")
        if agent.step in (Step.CLOSED_SUCCESS, Step.CLOSED_FAILURE):
            break

    # Show a quick observability summary for the session (latency, tool calls,
    # verification/payment outcomes, LLM vs rule-based).
    print()
    print(agent.metrics.render())


if __name__ == "__main__":
    main()
