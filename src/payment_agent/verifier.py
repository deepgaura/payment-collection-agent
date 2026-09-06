"""
WHAT THIS FILE IS (in one line):
    The "are you really you?" checker. It compares what the user CLAIMED
    against what the bank has ON FILE for that account.

THE RULE (from the assignment):
    A user passes ONLY if:
        the full name matches EXACTLY
        AND at least ONE of these also matches:
            - date of birth
            - last 4 digits of Aadhaar
            - pincode

    Example (account on file = Nithin Jain, dob 1990-05-14, aadhaar 4321):
        name "Nithin Jain" + dob "1990-05-14"   -> PASS (name + dob match)
        name "Nithin Jain" + aadhaar "4321"      -> PASS (name + aadhaar match)
        name "Nithin Raj"  + dob "1990-05-14"    -> FAIL (name is wrong)
        name "Nithin Jain" + dob "2000-01-01"    -> FAIL (no factor matches)

IMPORTANT SAFETY POINTS:
    - Matching is STRICT. "nithin jain" (lowercase) does NOT match "Nithin Jain".
      No fuzzy/approximate matching.
    - This is plain code (no LLM), so it always gives the same answer and can't
      be "talked around" by the user.
    - It NEVER returns the stored values, so it can't leak the real DOB/Aadhaar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # only for type hints; avoids an import loop at runtime
    from .state import Account, IdentityClaim


# The three "second proof" options a user can give (besides the name).
class SecondaryFactor(str, Enum):
    DOB = "dob"
    AADHAAR = "aadhaar_last4"
    PINCODE = "pincode"


@dataclass
class VerificationOutcome:
    """The result of a check.
        verified       = did they pass? (True/False)
        name_matches   = did the NAME match? (used to give better guidance)
        factor_matched = which factor matched, if any (dob/aadhaar/pincode)
    """
    verified: bool
    name_matches: bool
    factor_matched: SecondaryFactor | None = None


def _normalise_name(name: str) -> str:
    """
    Tidy a name for comparison WITHOUT being lenient:
      - trim spaces at the start/end
      - squeeze multiple inner spaces into one
    So "  Nithin   Jain " becomes "Nithin Jain".
    NOTE: we do NOT change the letters or their case - "nithin" stays "nithin".
    """
    return re.sub(r"\s+", " ", name.strip())


def name_matches(claimed: str | None, actual: str) -> bool:
    """
    Do the two names match exactly (after only the tidy-up above)?
      claimed = what the user typed,  actual = what's on file.
    "Nithin Jain" vs "Nithin Jain" -> True
    "nithin jain" vs "Nithin Jain" -> False (case differs -> strict fail)
    """
    if not claimed:
        return False   # they gave no name -> can't match
    return _normalise_name(claimed) == _normalise_name(actual)


def _digits(value: str | None) -> str | None:
    """
    Compare number-based factors (Aadhaar / pincode) by their DIGITS only.
      "4 3 2 1"  -> "4321"      (spaces are just formatting)
      "43-21"    -> "4321"
      "4321abc"  -> None        (has letters/junk -> we reject it, don't guess)
      None       -> None

    Why strip spaces? Because "4 3 2 1" and "4321" are the SAME number - that's
    how real ID systems compare them (Aadhaar is even printed in groups). This
    is NOT fuzzy matching; it's just ignoring formatting.
    """
    if value is None:
        return None
    # Only allow digits, spaces, and dashes. Anything else = reject.
    if not re.fullmatch(r"[\d\s\-]+", value):
        return None
    d = re.sub(r"\D", "", value)   # remove everything that isn't a digit
    return d or None


def verify(claim: "IdentityClaim", account: "Account") -> VerificationOutcome:
    """
    The main check. Compares the user's `claim` to the stored `account`.
    Returns a VerificationOutcome (never changes anything, never leaks data).
    """
    # STEP 1: The name MUST match. If it doesn't, stop right here - fail.
    nm = name_matches(claim.full_name, account.full_name)
    if not nm:
        return VerificationOutcome(verified=False, name_matches=False)

    # STEP 2: Name matched. Now we need AT LEAST ONE factor to also match.
    # Check each one; the first match means success.

    # (a) date of birth
    if claim.dob and claim.dob == account.dob:
        return VerificationOutcome(True, True, SecondaryFactor.DOB)

    # (b) Aadhaar last 4 (compared by digits only)
    if _digits(claim.aadhaar_last4) and _digits(claim.aadhaar_last4) == _digits(account.aadhaar_last4):
        return VerificationOutcome(True, True, SecondaryFactor.AADHAAR)

    # (c) pincode (compared by digits only)
    if _digits(claim.pincode) and _digits(claim.pincode) == _digits(account.pincode):
        return VerificationOutcome(True, True, SecondaryFactor.PINCODE)

    # STEP 3: Name was right, but NO factor matched -> not verified.
    # (name_matches=True lets the caller know the name itself was fine.)
    return VerificationOutcome(verified=False, name_matches=True)
