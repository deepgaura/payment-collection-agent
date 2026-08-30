"""Strict identity verification.

Rule (from the assignment):
    A user is verified if the full name matches EXACTLY, AND at least one of:
      - date of birth (YYYY-MM-DD)
      - last 4 digits of Aadhaar
      - pincode
    also matches.

Hard constraints honoured here:
- Matching is strict. No fuzzy matching. No case-insensitive name matching.
- The comparison happens entirely in deterministic code, never in the LLM,
  so it is reproducible and cannot be talked around by the user.
- The stored account values are never returned to the caller, so nothing here
  can leak DOB / Aadhaar / pincode back to the user.

Name normalisation policy: we compare the name exactly, only collapsing
internal whitespace runs to a single space and trimming leading/trailing
whitespace on BOTH sides. This treats "Nithin  Jain" (double space, a typing
artefact) the same as "Nithin Jain" but does NOT lower-case, transliterate, or
fuzzy-match. This is the single, explicitly-documented normalisation; case and
spelling must otherwise match exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid import cycle at runtime
    from .state import Account, IdentityClaim


class SecondaryFactor(str, Enum):
    DOB = "dob"
    AADHAAR = "aadhaar_last4"
    PINCODE = "pincode"


@dataclass
class VerificationOutcome:
    verified: bool
    name_matches: bool
    factor_matched: SecondaryFactor | None = None


def _normalise_name(name: str) -> str:
    """Trim ends and collapse internal whitespace. No case folding."""
    return re.sub(r"\s+", " ", name.strip())


def name_matches(claimed: str | None, actual: str) -> bool:
    if not claimed:
        return False
    return _normalise_name(claimed) == _normalise_name(actual)


def _digits(value: str | None) -> str | None:
    """Normalise a numeric factor for comparison.

    Real identity systems compare the *numeric value*, treating spaces/dashes as
    display formatting (Aadhaar is officially grouped as "1234 5678 9012"), so
    "4 3 2 1" == "4321". This is standard and expected - NOT fuzzy matching.

    Defence in depth: we only tolerate digits and the usual separators (space,
    dash). A value containing letters or other junk (e.g. "4321abc") is treated
    as no value rather than salvaging digits out of arbitrary text, so the
    verifier never relies on an upstream layer having sanitised the input.
    """
    if value is None:
        return None
    if not re.fullmatch(r"[\d\s\-]+", value):
        return None
    d = re.sub(r"\D", "", value)
    return d or None


def verify(claim: "IdentityClaim", account: "Account") -> VerificationOutcome:
    """Return whether the claim satisfies strict verification against account.

    Does not mutate anything and does not expose account values.
    """
    nm = name_matches(claim.full_name, account.full_name)
    if not nm:
        return VerificationOutcome(verified=False, name_matches=False)

    # Name matched; now require at least one strict secondary-factor match.
    if claim.dob and claim.dob == account.dob:
        return VerificationOutcome(True, True, SecondaryFactor.DOB)
    if _digits(claim.aadhaar_last4) and _digits(claim.aadhaar_last4) == _digits(account.aadhaar_last4):
        return VerificationOutcome(True, True, SecondaryFactor.AADHAAR)
    if _digits(claim.pincode) and _digits(claim.pincode) == _digits(account.pincode):
        return VerificationOutcome(True, True, SecondaryFactor.PINCODE)

    return VerificationOutcome(verified=False, name_matches=True)
