"""
WHAT THIS FILE IS (in one line):
    The shared "blank form" + the rules every translator (extractor) must follow.

BACKGROUND:
    An "extractor" (translator) reads the user's messy sentence and pulls out
    clean facts. This file does NOT do any translating itself. It just defines:
      1. Expecting        - a hint: "what are we collecting right now?"
      2. ExtractionResult - the blank form the translator fills in
      3. Extractor        - the rule that says "every translator must have an
                            extract() method"

    Both translators (rule_based.py and llm.py) share these definitions, so the
    rest of the code doesn't care which one ran - it always gets the same form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol


# ---------------------------------------------------------------------------
# 1) Expecting: "what is the agent collecting right now?"
# ---------------------------------------------------------------------------
# This hint helps the translator read ambiguous input correctly.
# Example: the text "500"
#   - if we're EXPECTING AMOUNT  -> it's ₹500 to pay
#   - if we're EXPECTING IDENTITY -> it's treated as an ID number, not money
class Expecting(str, Enum):
    ACCOUNT = "account"            # want the account ID (e.g. "ACC1001")
    IDENTITY = "identity"          # want name / DOB / Aadhaar / pincode
    AMOUNT = "amount"              # want the payment amount
    CARD = "card"                  # want card number / expiry / CVV / name
    CONFIRMATION = "confirmation"  # want a yes/no before charging


# ---------------------------------------------------------------------------
# 2) ExtractionResult: the "filled-in form" the translator returns
# ---------------------------------------------------------------------------
# Think of it as a form with many boxes. The translator fills in only the boxes
# it's confident about and leaves the rest as None (empty).
#
# Example: user says "the card number is 4532 0151 1283 0366"
#   -> card_number = "4532015112830366", everything else stays None.
#
# IMPORTANT: these are just *candidates* (guesses). Nothing here is trusted yet
# - the deterministic code later validates/verifies each value.
@dataclass
class ExtractionResult:
    account_id: Optional[str] = None
    full_name: Optional[str] = None
    dob_text: Optional[str] = None          # the RAW date text; parsed later, e.g. "14th May 1990"
    aadhaar_last4: Optional[str] = None      # 4 digits
    pincode: Optional[str] = None            # 6 digits
    amount: Optional[float] = None           # e.g. 1000.0
    pay_full_balance: bool = False           # True if user said "pay the full amount"

    cardholder_name: Optional[str] = None
    card_number: Optional[str] = None        # digits only
    cvv: Optional[str] = None                # digits only
    expiry_month: Optional[int] = None       # 1-12
    expiry_year: Optional[int] = None        # e.g. 2027

    # True if the user wants to cancel/stop ("quit", "cancel", "bye"...).
    wants_to_quit: bool = False

    # The user's yes/no at the "confirm before charging" step.
    #   True  = go ahead,  False = cancel,  None = unclear (agent re-asks, no charge)
    # The LLM/regex only DETECTS this; the code decides what to do with it.
    confirm: Optional[bool] = None

    # True if the user gave a number as their ID factor, but it wasn't a valid
    # 4-digit Aadhaar or 6-digit pincode (e.g. a 5-digit typo). Lets the agent
    # politely ASK again instead of silently ignoring it.
    unclear_secondary: bool = False

    # Bookkeeping (not shown to the user): which translator produced this
    # ("llm" or "rule_based"), and any notes for debugging/metrics.
    source: str = "rule_based"
    notes: list[str] = field(default_factory=list)

    def merge_missing_from(self, other: "ExtractionResult") -> None:
        """
        Combine two forms: keep what THIS form already has, and fill any EMPTY
        box using the OTHER form.

        We use this to merge the LLM's result with the rule-based result: the
        LLM leads, and regex fills in anything the LLM missed (or vice-versa).

        Example:
            self  (LLM):  {name: "Nithin Jain", cvv: None}
            other (regex):{name: None,          cvv: "123"}
            after merge:  {name: "Nithin Jain", cvv: "123"}   # gaps filled
        """
        # For each normal field: if mine is empty but theirs isn't, copy theirs.
        for name in (
            "account_id", "full_name", "dob_text", "aadhaar_last4", "pincode",
            "amount", "cardholder_name", "card_number", "cvv",
            "expiry_month", "expiry_year",
        ):
            if getattr(self, name) is None and getattr(other, name) is not None:
                setattr(self, name, getattr(other, name))
        # For the True/False flags: True wins (if either detected it, keep True).
        self.pay_full_balance = self.pay_full_balance or other.pay_full_balance
        self.wants_to_quit = self.wants_to_quit or other.wants_to_quit
        self.unclear_secondary = self.unclear_secondary or other.unclear_secondary
        # For confirm: only take theirs if I don't already have a yes/no.
        if self.confirm is None:
            self.confirm = other.confirm


# ---------------------------------------------------------------------------
# 3) Extractor: the "contract" every translator must follow
# ---------------------------------------------------------------------------
# This is a Protocol (like an interface): it just says "anything that wants to
# be an Extractor MUST have an extract(text, expecting) method that returns an
# ExtractionResult." Both RuleBasedExtractor and LLMExtractor satisfy this, so
# they're interchangeable.
class Extractor(Protocol):
    def extract(self, text: str, expecting: Expecting) -> ExtractionResult:
        """Read one user message and return the filled-in form."""
        ...
