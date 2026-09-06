"""
WHAT THIS FILE IS (in one line):
    The "notebook" - the data that remembers everything about one conversation.

IT'S JUST DATA (no decisions):
    This file only DEFINES the shapes we store. The brain (orchestrator.py)
    reads and writes these; agent.py owns one SessionState per chat. The LLM
    never touches this directly.

WHAT'S IN HERE:
    Step          - an enum of the conversation stages (greeting, verify, pay...)
    Account       - the real account info from the API (SENSITIVE - never shown)
    IdentityClaim - what the USER claims (compared against Account by verifier)
    CardDetails   - card fields, held only while building a payment (then wiped)
    SessionState  - the whole notebook: current step + all of the above + counters

SECURITY NOTE:
    Card number/CVV live only in `card` while assembling a payment and are wiped
    by clear_card() the moment the payment ends. The real DOB/Aadhaar/pincode
    live in `account` and are NEVER put into a message to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Step(str, Enum):
    """
    The stages of the conversation, in order. `SessionState.step` always holds
    exactly one of these - it's "which page of the form are we on".

    Using named steps (instead of loose strings) means the code can't
    accidentally jump to, say, payment before verification - the flow only moves
    one step at a time.
    """

    GREETING = "greeting"              # waiting to greet / for first message
    AWAIT_ACCOUNT = "await_account"    # asking for account id
    AWAIT_IDENTITY = "await_identity"  # collecting name + secondary factor
    AWAIT_AMOUNT = "await_amount"      # verified; collecting payment amount
    AWAIT_CARD = "await_card"          # collecting card details
    AWAIT_CONFIRMATION = "await_confirmation"  # card ready; awaiting user's go-ahead
    PROCESSING = "processing"          # about to call payment API
    CLOSED_SUCCESS = "closed_success"  # terminal: payment succeeded
    CLOSED_FAILURE = "closed_failure"  # terminal: gave up / unrecoverable


# The two "the conversation is over" steps. If we're on one of these, next()
# just restates the closing message.
TERMINAL_STEPS = frozenset({Step.CLOSED_SUCCESS, Step.CLOSED_FAILURE})


class MalformedAccountError(ValueError):
    """
    Thrown if the lookup API gives back a body we can't read into an Account
    (missing fields, weird balance). The orchestrator catches this and shows a
    graceful "try again" instead of crashing the turn.
    """


@dataclass
class Account:
    """
    The real account data from the lookup API. This is SENSITIVE - the verifier
    compares against it, but its values are never shown to the user.
    """

    account_id: str
    full_name: str
    dob: str
    aadhaar_last4: str
    pincode: str
    balance: float

    @classmethod
    def from_api(cls, data: dict) -> "Account":
        """
        Build an Account from the API's JSON, checking it has all the fields we
        need. If something's missing or the balance isn't a number, raise
        MalformedAccountError (one clean error type) instead of a random crash.
        """
        if not isinstance(data, dict):
            raise MalformedAccountError("lookup response was not an object")
        required = ("account_id", "full_name", "dob", "aadhaar_last4", "pincode", "balance")
        missing = [k for k in required if data.get(k) is None]
        if missing:
            raise MalformedAccountError(f"lookup response missing fields: {missing}")
        try:
            balance = float(data["balance"])
        except (TypeError, ValueError):
            raise MalformedAccountError("lookup response had a non-numeric balance")
        return cls(
            account_id=str(data["account_id"]),
            full_name=str(data["full_name"]),
            dob=str(data["dob"]),
            aadhaar_last4=str(data["aadhaar_last4"]),
            pincode=str(data["pincode"]),
            balance=balance,
        )


@dataclass
class IdentityClaim:
    """
    What the USER says about themselves during verification. These are just
    claims (not trusted) - the verifier compares them against the real Account.
    """

    full_name: Optional[str] = None
    dob: Optional[str] = None            # stored as YYYY-MM-DD once parsed
    aadhaar_last4: Optional[str] = None
    pincode: Optional[str] = None

    def has_secondary_factor(self) -> bool:
        # True if the user has given at least ONE of dob/aadhaar/pincode.
        # (Verification needs the name + at least one of these.)
        return any((self.dob, self.aadhaar_last4, self.pincode))


@dataclass
class CardDetails:
    """
    The card fields, held ONLY while we're building a payment. Wiped as soon as
    the payment finishes (success/cancel/failure). Never written to logs raw.
    """

    cardholder_name: Optional[str] = None
    card_number: Optional[str] = None       # digits only
    cvv: Optional[str] = None
    expiry_month: Optional[int] = None
    expiry_year: Optional[int] = None

    def is_complete(self) -> bool:
        # True only when ALL four card fields are filled in.
        return all(
            v is not None
            for v in (
                self.cardholder_name,
                self.card_number,
                self.cvv,
                self.expiry_month,
                self.expiry_year,
            )
        )

    def next_missing_field(self) -> Optional[str]:
        """
        Return the name of the NEXT card field we still need, in ask-order
        (number -> expiry -> cvv -> name), or None if the card is complete.
        The orchestrator uses this to ask for one field at a time.
        """
        if not self.card_number:
            return "card_number"
        if self.expiry_month is None or self.expiry_year is None:
            return "expiry"
        if not self.cvv:
            return "cvv"
        if not self.cardholder_name:
            return "cardholder_name"
        return None


@dataclass
class SessionState:
    """
    THE NOTEBOOK - everything we remember about one conversation. One of these
    is created per Agent and passed around so every part shares the same memory.
    """

    step: Step = Step.GREETING          # which stage we're on (starts at greeting)

    account: Optional[Account] = None   # filled in after a successful lookup

    claim: IdentityClaim = field(default_factory=IdentityClaim)  # what the user claimed
    is_verified: bool = False           # have they passed verification?

    # What they're paying:
    amount: Optional[float] = None
    card: CardDetails = field(default_factory=CardDetails)

    transaction_id: Optional[str] = None  # set once a payment succeeds

    # --- "how many tries" counters (for the retry limits) ---
    verification_attempts: int = 0
    payment_attempts: int = 0
    account_lookup_attempts: int = 0
    network_failures: int = 0
    # Turns in a row where nothing moved forward -> used to end a stuck chat.
    no_progress_turns: int = 0

    # --- one-turn scratchpad ---
    # Things the extractor found THIS turn that a handler will use in a moment.
    # (Kept as real fields so all state is visible in one place.)
    pending_account_id: Optional[str] = None   # account id waiting to be looked up
    pending_amount: Optional[float] = None      # amount waiting to be validated
    pending_pay_full: bool = False              # user said "pay the full amount"
    unclear_secondary: bool = False             # gave a number we couldn't classify

    def clear_card(self) -> None:
        """Wipe the card data (replace with a fresh empty one). Called whenever
        a payment ends - success, cancel, or failure."""
        self.card = CardDetails()

    def is_terminal(self) -> bool:
        # True if the conversation has ended (success or failure).
        return self.step in TERMINAL_STEPS
