"""
WHAT THIS FILE IS (in one line):
    The "is this input actually valid?" checkers - card number, CVV, expiry,
    amount, date, and account-id shape.

WHY WE CHECK LOCALLY (before calling the API):
    The real payment API often just says "invalid_args" without saying WHAT was
    wrong. So we check things ourselves FIRST, which lets us tell the user the
    exact problem ("that card has expired", "CVV should be 3 digits", etc.).
    The API is only a backup check.

WHO USES THIS FILE (the connections):
    - orchestrator.py calls these RIGHT BEFORE it charges the card:
        validate_card_number, validate_cvv, validate_expiry  -> check the card
        validate_amount                                      -> check the amount
      If any fails, the orchestrator shows the failure `reason` and re-asks.
    - orchestrator.py calls normalize_account_id when the user gives an account
      id (to reject typos like "AC1001" instead of guessing).
    - orchestrator.py calls parse_date_strict to turn "14th May 1990" into
      "1990-05-14" before the verifier compares it.
    - format_currency is used by responses.py / here to print "₹1,250.75".

HOW RESULTS ARE RETURNED:
    Every checker returns a `ValidationResult` (see below): either "ok + the
    cleaned value" or "not ok + a friendly reason to show the user". We return
    this instead of throwing errors, so the caller has no messy try/except.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    """
    The answer every checker gives back. Two possible shapes:
      PASS:  ok=True,  value=<the cleaned value>   (reason is empty)
      FAIL:  ok=False, reason=<friendly message>   (value is empty)

    Example (PASS): validate_amount(500, 1000) -> ok=True, value=500.0
    Example (FAIL): validate_amount(0, 1000)   -> ok=False,
                    reason="The amount must be greater than zero."
    """
    ok: bool
    value: Optional[object] = None   # the cleaned-up value, when it passed
    reason: Optional[str] = None     # the message to show the user, when it failed

    @classmethod
    def success(cls, value) -> "ValidationResult":
        # Shortcut to build a PASS result.
        return cls(ok=True, value=value)

    @classmethod
    def failure(cls, reason: str) -> "ValidationResult":
        # Shortcut to build a FAIL result with a message for the user.
        return cls(ok=False, reason=reason)


# --------------------------------------------------------------------------- #
# ACCOUNT ID  (used by orchestrator when the user gives their account number)
# --------------------------------------------------------------------------- #
# A valid id is the letters "ACC" then digits. We tidy case/spaces but we NEVER
# "fix" a wrong one - because the id decides WHICH account we check against.
# Guessing could pull up the wrong person's account, so a bad id is rejected.
_ACCOUNT_ID_RE = re.compile(r"^\s*acc\s*([0-9]+)\s*$", re.I)


def normalize_account_id(raw: Optional[str]) -> Optional[str]:
    """
    Turn the user's account id into the clean "ACC<digits>" form, or None if it
    isn't a proper account id.
        "acc 1001" -> "ACC1001"      (just tidied)
        "ACC1001"  -> "ACC1001"
        "AC1001"   -> None           (one C -> NOT repaired, rejected)
        "1001"     -> None           (no "ACC" -> rejected)
    """
    if not raw:
        return None
    m = _ACCOUNT_ID_RE.match(str(raw))
    if not m:
        return None
    return "ACC" + m.group(1)

# check if the card number has the correct mathematical structur or not - luhn test 
# --------------------------------------------------------------------------- #
# CARD NUMBER  (used by orchestrator before charging)
# --------------------------------------------------------------------------- #
def luhn_ok(number: str) -> bool:
    """
    The "Luhn check" - a simple math test that every real card number passes
    and most typos fail. It's how we catch a mistyped card without asking a bank.
    Returns True if the number passes the check.
    (You don't need to memorise the math; just know: pass = looks like a real
    card number, fail = almost certainly a typo.)
    """
    if not number.isdigit():
        return False
    total = 0
    reverse = number[::-1]                 # read the digits right-to-left
    for i, ch in enumerate(reverse):
        d = int(ch)
        if i % 2 == 1:                     # every 2nd digit: double it...
            d *= 2
            if d > 9:                      # ...and if it's >9, subtract 9
                d -= 9
        total += d
    return total % 10 == 0                 # valid if the total is a multiple of 10


def validate_card_number(raw: Optional[str]) -> ValidationResult:
    """
    Check the card number the user gave. Returns PASS with the cleaned digits,
    or FAIL with a reason. Called by orchestrator._handle_card before payment.
        "4532 0151 1283 0366" -> PASS, value "4532015112830366"
        "1234 5678 9012 3456" -> FAIL (fails the Luhn check)
        "4532 **** **** 0366" -> FAIL (masked)
    """
    if not raw:
        return ValidationResult.failure("I still need your card number.")
    digits = re.sub(r"\D", "", raw)        # keep only the digits
    # Reject masked numbers like "4532 **** **** 0366".
    if "*" in raw or "x" in raw.lower():
        return ValidationResult.failure(
            "That card number looks masked. Please share the full number."
        )
    # A real card number is 12-19 digits long.
    if not (12 <= len(digits) <= 19):
        return ValidationResult.failure(
            "That card number doesn't look right - it should be 13-19 digits. "
            "Could you re-enter it?"
        )
    # Run the math check to catch typos.
    if not luhn_ok(digits):
        return ValidationResult.failure(
            "That card number didn't pass validation. Please double-check the "
            "digits and re-enter it."
        )
    return ValidationResult.success(digits)


def is_amex(card_number: str) -> bool:
    # Amex cards start with 34 or 37, and their CVV is 4 digits (not 3).
    # Used by validate_cvv to know how many CVV digits to expect.
    return card_number.startswith(("34", "37"))


# --------------------------------------------------------------------------- #
# CVV  (used by orchestrator before charging)
# --------------------------------------------------------------------------- #
def validate_cvv(raw: Optional[str], card_number: Optional[str]) -> ValidationResult:
    """
    Check the CVV. It must be 3 digits for most cards, 4 for Amex. We pass in
    the card number so we know which length to expect.
        "123"  with a Visa card   -> PASS
        "12"   with a Visa card    -> FAIL (too short)
        "1234" with an Amex card   -> PASS
    """
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
# EXPIRY  (used by orchestrator before charging)
# --------------------------------------------------------------------------- #
def validate_expiry(
    month: Optional[int],
    year: Optional[int],
    *,
    today: Optional[_dt.date] = None,   # only used by tests to fake "today"
) -> ValidationResult:
    """
    Check the card's expiry month+year. Must be a real month, and the card must
    not already be expired.
        (12, 2027) today=2026  -> PASS
        (1, 2020)  today=2026  -> FAIL (expired)
        (13, 2027)             -> FAIL (no month 13)
        (12, 27)               -> PASS, treated as year 2027
    """
    if month is None or year is None:
        return ValidationResult.failure("I still need the card's expiry month and year.")
    if not (1 <= month <= 12):
        return ValidationResult.failure(
            "That expiry month is invalid. Please give a month between 1 and 12."
        )
    # Turn a short year like 27 into 2027.
    if year < 100:
        year += 2000
    if not (2000 <= year <= 2099):
        return ValidationResult.failure("That expiry year doesn't look right.")

    today = today or _dt.date.today()
    # A card is good until the LAST day of its expiry month. So we compare
    # "today" against the end of that month.
    last_day = calendar.monthrange(year, month)[1]   # e.g. 31 for December
    expiry_end = _dt.date(year, month, last_day)
    if expiry_end < today:
        return ValidationResult.failure(
            "That card appears to have expired. Please use a card that is still valid."
        )
    return ValidationResult.success((month, year))


# --------------------------------------------------------------------------- #
# AMOUNT  (used by orchestrator when collecting how much to pay)
# --------------------------------------------------------------------------- #
def validate_amount(raw_amount: Optional[float], balance: float) -> ValidationResult:
    """
    Check the payment amount. Rules:
      - must be more than 0
      - at most 2 decimal places (money can't have 3)
      - can't be more than what they owe (`balance`); paying less is fine

    Examples (balance = 1000):
        500     -> PASS (partial payment)
        1000    -> PASS (pays it all)
        0       -> FAIL (must be > 0)
        10.999  -> FAIL (3 decimals)
        2000    -> FAIL (more than they owe)
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
    # "round to 2 decimals and see if it changed" = a quick way to reject
    # things like 10.999 (more than 2 decimal places).
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
# DATE OF BIRTH  (used by orchestrator to turn a spoken date into YYYY-MM-DD,
#                 which the verifier then compares against the account)
# --------------------------------------------------------------------------- #
# All the ways people write a month name, mapped to its number (May -> 5).
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def _valid_calendar_date(y: int, m: int, d: int) -> Optional[str]:
    """
    Is (year, month, day) a REAL date on the calendar? If yes, return it as
    "YYYY-MM-DD"; if not, return None.
        1988, 2, 29 -> "1988-02-29"  (1988 was a leap year, so Feb 29 exists)
        1989, 2, 29 -> None          (1989 is NOT a leap year)
        1990, 2, 30 -> None          (Feb never has 30 days)
    We let Python's date() do the checking - it knows the calendar rules.
    """
    try:
        return _dt.date(y, m, d).isoformat()
    except ValueError:
        return None


def parse_date_strict(text: str) -> ValidationResult:
    """
    Read a date of birth written in ANY common style and return it in the
    standard "YYYY-MM-DD" form. If it's not a real/readable date, FAIL.
        "1990-05-14"           -> "1990-05-14"
        "14-05-1990"           -> "1990-05-14"
        "I was born on 14th May 1990" -> "1990-05-14"
        "May 14, 90"           -> "1990-05-14"
        "30th February"        -> FAIL (not a real date)

    IMPORTANT: this only fixes the FORMAT. It does NOT check if the date is the
    "right" one - that's the verifier's job later. So a wrong-but-real date
    (e.g. a nearby birthday) parses fine here and simply fails the match later.
    """
    if not text:
        return ValidationResult.failure("Could you share your date of birth?")
    t = text.strip().lower()

    # TRY 1: numbers with separators, YEAR first -> "1990-05-14" / "1990/5/14".
    m = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", t)
    if m:
        iso = _valid_calendar_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if iso:
            return ValidationResult.success(iso)

    # TRY 2: numbers with separators, DAY first -> "14-05-1990" / "14/5/90".
    m = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b", t)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:                      # turn "90" into "1990"
            year += 1900 if year > 30 else 2000
        iso = _valid_calendar_date(year, month, day)
        if iso:
            return ValidationResult.success(iso)

    # TRY 3: a written-out month, like "14th May 1990" or "May 14, 90".
    # First, find which month name appears.
    month_num = None
    for name, num in _MONTHS.items():
        if re.search(rf"\b{name}\b", t):
            month_num = num
            break
    if month_num is not None:
        # Then grab the numbers and figure out which is the day and the year.
        nums = re.findall(r"\d{1,4}", t)
        day = year = None
        for n in nums:
            v = int(n)
            if len(n) >= 3 or v > 31:            # 3+ digits or >31 = it's a year
                year = v if v > 100 else 1900 + v
            elif 1 <= v <= 31 and day is None:   # 1-31 = it's the day
                day = v
        # "May 14, 90": two small numbers -> the last one is the year.
        if year is None and len(nums) >= 2:
            yr = int(nums[-1])
            year = yr + (1900 if yr > 30 else 2000)
        if day and year:
            iso = _valid_calendar_date(year, month_num, day)
            if iso:
                return ValidationResult.success(iso)

    # Couldn't make sense of it -> ask again.
    return ValidationResult.failure(
        "I couldn't read that as a date. Could you share your date of birth, "
        "for example 1990-05-14?"
    )


# --------------------------------------------------------------------------- #
# MONEY FORMATTING  (used here and by responses.py to show amounts nicely)
# --------------------------------------------------------------------------- #
def format_currency(amount: float) -> str:
    """
    Turn a plain number into a rupee string with commas and 2 decimals.
        1250.75 -> "₹1,250.75"
        500     -> "₹500.00"
    (\\u20b9 is just the ₹ symbol.)
    """
    return f"\u20b9{amount:,.2f}"
