"""Extractor interface and the structured result it produces.

The extractor is the ONLY place the LLM is (optionally) used. Its single job:
read the user's raw message plus a small hint about what the agent is currently
expecting, and return structured candidate fields. It makes NO decisions about
verification, validation, payment, or flow - those belong to deterministic code.

Returning candidates (not commands) keeps the trust boundary clean: whatever
the extractor produces is treated as an untrusted *claim* and is independently
validated/verified downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol


class Expecting(str, Enum):
    """A hint to the extractor about the current conversational context.

    This focuses extraction (e.g. when we're expecting an amount, "500" is an
    amount, not a pincode) while still allowing out-of-order data capture.
    """

    ACCOUNT = "account"
    IDENTITY = "identity"
    AMOUNT = "amount"
    CARD = "card"


@dataclass
class ExtractionResult:
    """Structured candidate fields parsed from one user message.

    Every field is Optional; the extractor only sets what it is confident about.
    Fields are raw candidates - downstream validation decides if they're usable.
    """

    account_id: Optional[str] = None
    full_name: Optional[str] = None
    dob_text: Optional[str] = None          # raw date text; parsed strictly later
    aadhaar_last4: Optional[str] = None
    pincode: Optional[str] = None
    amount: Optional[float] = None
    pay_full_balance: bool = False          # user asked to "clear the full amount"

    cardholder_name: Optional[str] = None
    card_number: Optional[str] = None
    cvv: Optional[str] = None
    expiry_month: Optional[int] = None
    expiry_year: Optional[int] = None

    # Free-form intents that affect flow (handled deterministically upstream).
    wants_to_quit: bool = False

    # Set when the user clearly tried to give a secondary factor (a digit run)
    # at the identity step but it could not be confidently classified as a
    # 4-digit Aadhaar or 6-digit pincode. Lets the agent ASK instead of
    # silently dropping the value. False by default so other paths are unaffected.
    unclear_secondary: bool = False

    # Provenance, useful for eval/observability. Not shown to users.
    source: str = "rule_based"
    notes: list[str] = field(default_factory=list)

    def merge_missing_from(self, other: "ExtractionResult") -> None:
        """Fill any field that is unset on self using values from other.

        Used to combine LLM output with rule-based output: the LLM is primary,
        deterministic regex fills gaps (and vice-versa)."""
        for name in (
            "account_id", "full_name", "dob_text", "aadhaar_last4", "pincode",
            "amount", "cardholder_name", "card_number", "cvv",
            "expiry_month", "expiry_year",
        ):
            if getattr(self, name) is None and getattr(other, name) is not None:
                setattr(self, name, getattr(other, name))
        self.pay_full_balance = self.pay_full_balance or other.pay_full_balance
        self.wants_to_quit = self.wants_to_quit or other.wants_to_quit
        self.unclear_secondary = self.unclear_secondary or other.unclear_secondary


class Extractor(Protocol):
    def extract(self, text: str, expecting: Expecting) -> ExtractionResult:
        """Parse one user message into structured candidates."""
        ...
