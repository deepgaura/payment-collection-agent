"""Payment Collection AI Agent.

A conversational agent that handles an end-to-end payment collection flow:
greet -> look up account -> verify identity -> share balance -> collect
payment -> process payment -> recap and close.

Design principle: the LLM is used exclusively as a natural-language
understanding (extraction) layer that turns messy free-form user input into
structured candidate fields. All control flow, verification, validation, retry
accounting, and API calls are deterministic Python. This keeps the security
critical logic reproducible and impossible to bypass via prompt injection.
"""

from .agent import Agent

__all__ = ["Agent"]
__version__ = "1.0.0"
