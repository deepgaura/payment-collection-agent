"""Conversation state.

A single `SessionState` instance holds everything the agent needs to know
about one conversation. It is owned by the `Agent` object and mutated only by
the deterministic orchestrator. The LLM never touches this directly.

Security note: card data (PAN/CVV) is held only transiently in `card` while a
payment is being assembled and is wiped via `clear_card()` as soon as a
terminal payment outcome is reached. Sensitive account fields (dob, aadhaar,
pincode) live in `account` and are NEVER rendered back to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Step(str, Enum):
    """The stages of the collection flow, in order.

    Using an explicit enum (rather than free-form strings) makes the state
    machine auditable and prevents illegal transitions such as jumping to
    payment before verification.
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


# Steps after which the conversation is over and next() should just restate the
# closing message.
TERMINAL_STEPS = frozenset({Step.CLOSED_SUCCESS, Step.CLOSED_FAILURE})


class MalformedAccountError(ValueError):
    """Raised when the lookup API returns a body we can't parse into an Account.

    The orchestrator catches this and degrades gracefully instead of letting it
    crash the turn (the `next()` contract must never raise)."""


@dataclass
class Account:
    """Account data returned by the lookup API. Treated as sensitive."""

    account_id: str
    full_name: str
    dob: str
    aadhaar_last4: str
    pincode: str
    balance: float

    @classmethod
    def from_api(cls, data: dict) -> "Account":
        """Build an Account from a lookup response, validating shape.

        Raises MalformedAccountError (never a bare KeyError/ValueError) if a
        required field is missing or the balance isn't numeric, so callers have
        a single, typed failure to handle."""
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
    """What the *user* has claimed so far during verification.

    These are compared against `Account` by the verifier. They are the user's
    assertions, not trusted facts.
    """

    full_name: Optional[str] = None
    dob: Optional[str] = None            # normalised to YYYY-MM-DD when parsed
    aadhaar_last4: Optional[str] = None
    pincode: Optional[str] = None

    def has_secondary_factor(self) -> bool:
        return any((self.dob, self.aadhaar_last4, self.pincode))


@dataclass
class CardDetails:
    """Transient card data. Cleared as soon as payment reaches a terminal
    outcome. Never logged in raw form."""

    cardholder_name: Optional[str] = None
    card_number: Optional[str] = None       # digits only
    cvv: Optional[str] = None
    expiry_month: Optional[int] = None
    expiry_year: Optional[int] = None

    def is_complete(self) -> bool:
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
        """Return the FIRST still-missing card field, in the order we ask for
        them (number -> expiry -> cvv -> name), or None if complete.

        Collecting one field at a time gives a natural, call-centre-like flow.
        We still accept multiple fields in one message (greedy capture upstream);
        this only decides which single field to *prompt* for next.
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
    """The complete state of one conversation."""

    step: Step = Step.GREETING

    # Populated after a successful account lookup.
    account: Optional[Account] = None

    # The user's identity claims (compared against `account`).
    claim: IdentityClaim = field(default_factory=IdentityClaim)
    is_verified: bool = False

    # Payment intent.
    amount: Optional[float] = None
    card: CardDetails = field(default_factory=CardDetails)

    # Outcome.
    transaction_id: Optional[str] = None

    # Retry accounting.
    verification_attempts: int = 0
    payment_attempts: int = 0
    account_lookup_attempts: int = 0
    network_failures: int = 0
    # Consecutive turns that made no progress (no step change, no new usable
    # data). Bounds conversations where the user never supplies what's needed
    # so the agent can't loop forever.
    no_progress_turns: int = 0

    # Per-turn scratch: values extracted this turn that a step handler consumes.
    # Kept as declared fields (not ad-hoc attributes) so all state is explicit.
    pending_account_id: Optional[str] = None
    pending_amount: Optional[float] = None
    pending_pay_full: bool = False
    unclear_secondary: bool = False

    def clear_card(self) -> None:
        """Wipe transient card data. Called on any terminal payment outcome."""
        self.card = CardDetails()

    def is_terminal(self) -> bool:
        return self.step in TERMINAL_STEPS
