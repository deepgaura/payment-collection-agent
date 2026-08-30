"""Deterministic validators.

These run *before* any API call so the agent can give precise, actionable
feedback locally. This matters because the live API collapses several distinct
card problems into a single generic `invalid_args` error, so relying on the
server for user-facing reasons would produce vague messages. Client-side
validation is therefore the primary source of truth for "why did this fail",
and the API is a defensive backstop.

Every function returns a `ValidationResult` (ok flag + normalised value +
human-readable reason) rather than raising, so the orchestrator can compose
guidance without try/except noise.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    ok: bool
    value: Optional[object] = None   # normalised value when ok
    reason: Optional[str] = None     # user-facing message when not ok

    @classmethod
    def success(cls, value) -> "ValidationResult":
        return cls(ok=True, value=value)

    @classmethod
    def failure(cls, reason: str) -> "ValidationResult":
        return cls(ok=False, reason=reason)


# --------------------------------------------------------------------------- #
# Card number (Luhn)
# --------------------------------------------------------------------------- #
def luhn_ok(number: str) -> bool:
    """Standard Luhn checksum. Expects digits only."""
    if not number.isdigit():
        return False
    total = 0
    reverse = number[::-1]
    for i, ch in enumerate(reverse):
        d = int(ch)
        if i % 2 == 1:  # every second digit from the right
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def validate_card_number(raw: Optional[str]) -> ValidationResult:
    if not raw:
        return ValidationResult.failure("I still need your card number.")
    digits = re.sub(r"\D", "", raw)
    if "*" in raw or "x" in raw.lower():
        return ValidationResult.failure(
            "That card number looks masked. Please share the full number."
        )
    if not (12 <= len(digits) <= 19):
        return ValidationResult.failure(
            "That card number doesn't look right - it should be 13-19 digits. "
            "Could you re-enter it?"
        )
    if not luhn_ok(digits):
        return ValidationResult.failure(
            "That card number didn't pass validation. Please double-check the "
            "digits and re-enter it."
        )
    return ValidationResult.success(digits)


def is_amex(card_number: str) -> bool:
    return card_number.startswith(("34", "37"))


# --------------------------------------------------------------------------- #
# CVV
# --------------------------------------------------------------------------- #
def validate_cvv(raw: Optional[str], card_number: Optional[str]) -> ValidationResult:
    if not raw:
        return ValidationResult.failure("I still need the CVV from the back of your card.")
    digits = re.sub(r"\D", "", raw)
    expected = 4 if (card_number and is_amex(card_number)) else 3
    if len(digits) != expected:
        return ValidationResult.failure(
            f"The CVV should be {expected} digits for this card. Please re-enter it."
        )
    return ValidationResult.success(digits)


# --------------------------------------------------------------------------- #
# Expiry
# --------------------------------------------------------------------------- #
def validate_expiry(
    month: Optional[int],
    year: Optional[int],
    *,
    today: Optional[_dt.date] = None,
) -> ValidationResult:
    if month is None or year is None:
        return ValidationResult.failure("I still need the card's expiry month and year.")
    if not (1 <= month <= 12):
        return ValidationResult.failure(
            "That expiry month is invalid. Please give a month between 1 and 12."
        )
    # Normalise 2-digit years (e.g. 27 -> 2027).
    if year < 100:
        year += 2000
    if not (2000 <= year <= 2099):
        return ValidationResult.failure("That expiry year doesn't look right.")

    today = today or _dt.date.today()
    # A card is valid through the last day of its expiry month.
    last_day = calendar.monthrange(year, month)[1]
    expiry_end = _dt.date(year, month, last_day)
    if expiry_end < today:
        return ValidationResult.failure(
            "That card appears to have expired. Please use a card that is still valid."
        )
    return ValidationResult.success((month, year))


# --------------------------------------------------------------------------- #
# Amount
# --------------------------------------------------------------------------- #
def validate_amount(raw_amount: Optional[float], balance: float) -> ValidationResult:
    """Validate a payment amount against the documented rules.

    - must be > 0
    - at most 2 decimal places
    - must not exceed the outstanding balance (partial payments allowed)
    """
    if raw_amount is None:
        return ValidationResult.failure("How much would you like to pay?")
    try:
        amount = float(raw_amount)
    except (TypeError, ValueError):
        return ValidationResult.failure(
            "I couldn't read that as an amount. Please give a number, e.g. 500."
        )
    if amount <= 0:
        return ValidationResult.failure("The amount must be greater than zero.")
    # Guard against float noise: round to 2dp and compare.
    if round(amount, 2) != amount:
        return ValidationResult.failure(
            "Amounts can have at most two decimal places. Please re-enter it."
        )
    if round(amount, 2) > round(balance, 2):
        return ValidationResult.failure(
            f"That exceeds your outstanding balance of {format_currency(balance)}. "
            f"Please enter an amount up to {format_currency(balance)}."
        )
    return ValidationResult.success(round(amount, 2))


# --------------------------------------------------------------------------- #
# Strict date parsing (for DOB verification)
# --------------------------------------------------------------------------- #
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _valid_calendar_date(y: int, m: int, d: int) -> Optional[str]:
    """Return YYYY-MM-DD if (y,m,d) is a real calendar date, else None.

    This correctly accepts leap-year dates like 1988-02-29 and rejects
    non-existent ones like 1989-02-29 or 1990-02-30.
    """
    try:
        return _dt.date(y, m, d).isoformat()
    except ValueError:
        return None


def parse_date_strict(text: str) -> ValidationResult:
    """Parse a date-of-birth from free-form text into a canonical YYYY-MM-DD.

    Deliberately conservative: it only normalises the *format*. Whether the
    parsed date matches the account is decided later by strict equality, so a
    valid-but-wrong date (e.g. a nearby date) will parse fine and simply fail
    the match. Genuinely invalid dates (Feb 30) are rejected here.
    """
    if not text:
        return ValidationResult.failure("Could you share your date of birth?")
    t = text.strip().lower()

    # 1) ISO / numeric with separators: YYYY-MM-DD, DD-MM-YYYY, DD/MM/YYYY
    m = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", t)
    if m:
        iso = _valid_calendar_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if iso:
            return ValidationResult.success(iso)

    m = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b", t)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 1900 if year > 30 else 2000
        iso = _valid_calendar_date(year, month, day)
        if iso:
            return ValidationResult.success(iso)

    # 2) Textual month: "14th May 1990", "May 14, 90", "14 May 1990"
    month_num = None
    for name, num in _MONTHS.items():
        if re.search(rf"\b{name}\b", t):
            month_num = num
            break
    if month_num is not None:
        nums = re.findall(r"\d{1,4}", t)
        day = year = None
        for n in nums:
            v = int(n)
            if len(n) >= 3 or v > 31:            # looks like a year
                year = v if v > 100 else 1900 + v
            elif 1 <= v <= 31 and day is None:   # looks like a day
                day = v
        # Two 2-digit numbers e.g. "May 14, 90": second is the year.
        if year is None and len(nums) >= 2:
            yr = int(nums[-1])
            year = yr + (1900 if yr > 30 else 2000)
        if day and year:
            iso = _valid_calendar_date(year, month_num, day)
            if iso:
                return ValidationResult.success(iso)

    return ValidationResult.failure(
        "I couldn't read that as a date. Could you share your date of birth, "
        "for example 1990-05-14?"
    )


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def format_currency(amount: float) -> str:
    """Format an amount as Indian Rupees with thousands separators."""
    return f"\u20b9{amount:,.2f}"
