"""
WHAT THIS FILE IS (in one line):
    The "no-AI" translator: it uses text patterns (regex) and simple rules to
    pull clean fields out of the user's sentence.

WHEN IT'S USED:
    - As the BACKUP whenever the LLM is off or fails (so the app always works).
    - As the ONLY translator in tests (because it's 100% predictable - same
      input always gives the same output).

THE KEY TRICK - the `expecting` hint:
    The same text means different things depending on what we're collecting:
      - "400001" while collecting IDENTITY  -> a pincode
      - "500"    while collecting AMOUNT    -> ₹500 to pay
    So we only look for card fields when expecting CARD, only amounts when
    expecting AMOUNT, etc. This stops mix-ups.

HOW TO READ THIS FILE:
    `extract()` is the entry point. Based on `expecting`, it calls a small
    helper (`_identity`, `_amount`, `_card`, `_confirm`). Each helper uses
    regex to find its fields. Don't worry about the exact regex - the comment
    above each one says in plain words what it catches, with an example.
"""

from __future__ import annotations

import re
from typing import Optional

from .base import Expecting, ExtractionResult

# --------------------------------------------------------------------------- #
# Small lookup tables / patterns used by the helpers below.
# --------------------------------------------------------------------------- #

# Words that mean "I want to stop". If any appear, we flag wants_to_quit.
_QUIT_WORDS = re.compile(r"\b(quit|exit|cancel|stop|nevermind|never mind|bye|goodbye)\b", re.I)

# Common conversational filler / function words that should never appear inside
# a bare name. If any token is one of these, the phrase is treated as chatter,
# not a name. Kept small and high-precision on purpose.
_NAME_STOPWORDS = {
    "tell", "me", "a", "an", "the", "joke", "hmm", "ok", "okay", "yes", "no",
    "what", "when", "where", "why", "how", "who", "you", "your", "need", "again",
    "please", "hi", "hello", "hey", "thanks", "thank", "do", "does", "is", "are",
    "am", "was", "were", "will", "can", "could", "would", "should", "help",
    "want", "know", "dunno", "sure", "maybe", "and", "or", "but", "of", "to",
    "for", "with", "give", "get", "let", "us", "it", "this", "that", "here",
    "there", "now", "then", "pay", "paid", "not",
}

# Spoken single digits -> numbers. Used to read "one two three" as 1,2,3.
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
# Spoken multipliers, so "a thousand" = 1000, "two lakh" = 200000, etc.
_MULTIPLIERS = {
    "hundred": 100, "thousand": 1000, "lakh": 100000, "lakhs": 100000,
    "k": 1000, "million": 1000000,
}


def _words_to_digits(text: str) -> str:
    """Turn spoken digits into a digit string.
    Example: "one two three" -> "123". Handy for CVVs/pincodes read aloud."""
    out = []
    for tok in re.findall(r"[a-z]+|\d", text.lower()):
        if tok in _NUMBER_WORDS:
            out.append(str(_NUMBER_WORDS[tok]))
        elif tok.isdigit():
            out.append(tok)
    return "".join(out)


class RuleBasedExtractor:
    source = "rule_based"   # stamped onto results so we know regex produced them

    def extract(self, text: str, expecting: Expecting) -> ExtractionResult:
        # Start with an empty form; we'll fill in whatever we find.
        result = ExtractionResult(source=self.source)
        if not text:
            return result   # nothing typed -> empty form

        # Did the user ask to stop? (checked no matter what step we're on)
        if _QUIT_WORDS.search(text):
            result.wants_to_quit = True

        # Account id can appear at any time (out-of-order capture).

        acc = self._account_id(text)
        if acc:
            result.account_id = acc

        # Step-specific parsing, based on what we're currently collecting.
        if expecting == Expecting.IDENTITY:
            self._identity(text, result)      # name / dob / aadhaar / pincode
        elif expecting == Expecting.AMOUNT:
            self._amount(text, result)        # payment amount
        elif expecting == Expecting.CARD:
            self._card(text, result)          # card number / expiry / cvv / name
        elif expecting == Expecting.CONFIRMATION:
            result.confirm = self._confirm(text)   # yes / no
        elif expecting == Expecting.ACCOUNT:
            pass                              # account id already handled above

        return result

    # --- confirmation (yes / no) ------------------------------------------- #
    @staticmethod
    def _confirm(text: str) -> Optional[bool]:
        """Detect an explicit yes/no. Returns None if ambiguous (caller re-asks
        and, crucially, does NOT charge on ambiguity)."""
        t = text.strip().lower()
        # A "no" cue anywhere wins (safer default: don't charge if they object).
        if re.search(r"\b(no|nope|cancel|stop|wait|don'?t|do not|change|hold on|nah)\b", t):
            return False
        if re.search(r"\b(yes|yeah|yep|yup|sure|ok|okay|confirm|confirmed|proceed|go ahead|go|pay|do it|correct|right|please do)\b", t):
            return True
        return None

    # --- account ----------------------------------------------------------- #
    @staticmethod
    def _account_id(text: str) -> Optional[str]:
        # Find "acc" followed by digits and return it as "ACC<digits>".
        # Catches: "ACC1001", "acc 1001", "acc-1001", "account id: acc1001".
        # (Note: needs the letters "acc"; a bare "1001" is NOT an account id.)
        m = re.search(r"\bacc\W*?(\d{3,})\b", text, re.I)
        if m:
            return "ACC" + m.group(1)
        return None

    # --- identity ---------------------------------------------------------- #
    # Pulls out the four identity things: date of birth, aadhaar last-4,
    # pincode, and full name. Fills whichever it can find into `result`.
    def _identity(self, text: str, result: ExtractionResult) -> None:
        # DOB: we don't parse the date here - we just grab the raw phrase (e.g.
        # "14th May 1990") and let the strict date parser handle it later.
        result.dob_text = self._dob_text(text)

        # Aadhaar (4 digits) vs pincode (6 digits). First try to find them when
        # the user LABELS them, e.g. "aadhaar 4321" or "pincode 400001".
        low = text.lower()
        joined = _words_to_digits(text)  # e.g. "4 0 0 0 0 1" -> "400001"

        aadhaar = self._labelled_digits(text, r"aadhaar|aadhar|adhaar", 4)
        if aadhaar:
            result.aadhaar_last4 = aadhaar
        pincode = self._labelled_digits(text, r"pin\s*code|pincode|pin|zip", 6)
        if pincode:
            result.pincode = pincode

        # If they gave a number with NO label, guess by length: 6 = pincode,
        # 4 = aadhaar. (Length is the only clue we have here.)
        if not result.aadhaar_last4 and not result.pincode:
            for run in re.findall(r"\d(?:[\s-]*\d){2,}", text):
                digits = re.sub(r"\D", "", run)
                if len(digits) == 6:
                    result.pincode = digits
                elif len(digits) == 4:
                    result.aadhaar_last4 = digits
            # spoken-digit fallback ("4 0 0 0 0 1")
            if not result.pincode and len(joined) == 6:
                result.pincode = joined
            if not result.aadhaar_last4 and len(joined) == 4:
                result.aadhaar_last4 = joined

        # If the user clearly typed a number as their factor but it did not
        # resolve to a valid Aadhaar (4) or pincode (6) - and it's not a date -
        # flag it so the agent can ASK rather than silently ignore it.
        if (
            not result.aadhaar_last4
            and not result.pincode
            and result.dob_text is None
            and len(joined) >= 2
        ):
            result.unclear_secondary = True

        # Name: explicit "my name is X" / "call me ... my full name is X",
        # else a capitalised two+ word span with no digits.
        result.full_name = self._name(text)

    @staticmethod
    def _dob_text(text: str) -> Optional[str]:
        # Does this text look like it contains a date? If so, return the whole
        # text (the strict parser will pull the real date out later). Catches
        # "1990-05-14", "14/05/1990", and "14th May 1990".
        # numeric date
        if re.search(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", text) or re.search(
            r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}", text
        ):
            return text
        # textual month present + a number => likely a date phrase
        if re.search(
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", text, re.I
        ) and re.search(r"\d", text):
            return text
        return None

    @staticmethod
    def _labelled_digits(text: str, label_re: str, count: int) -> Optional[str]:
        """
        Find exactly `count` digits sitting next to a label word.
        Example: _labelled_digits("pincode 4 0 0 0 0 1", "pincode", 6) -> "400001".
        Also understands "ends with 9876" / "last four 4321" phrasing.
        """
        m = re.search(
            rf"(?:{label_re})[^0-9]*((?:\d[\s-]*){{{count}}})", text, re.I
        )
        if m:
            digits = re.sub(r"\D", "", m.group(1))
            if len(digits) == count:
                return digits
        # "ends with 9876" style
        m2 = re.search(rf"(?:ends?\s+with|last\s+four|last\s+4)\D*(\d{{{count}}})", text, re.I)
        if m2:
            return m2.group(1)
        return None

    @staticmethod
    def _name(text: str) -> Optional[str]:
        # Try to pull out a person's name, in order of confidence:
        #
        # 1) "full name is X"  -> best; wins even over a nickname.
        #    ("call me Raja but my full name is Rajarajeswari" -> Rajarajeswari)
        m = re.search(r"full name is\s+([A-Za-z][A-Za-z .'-]+)", text, re.I)
        if m:
            return _clean_name(m.group(1))
        # 2) "my name is X" / "i'm X" / "it's X".
        m = re.search(r"\b(?:my name is|name is|i am|i'm|this is|it's)\s+([A-Za-z][A-Za-z .'-]+)", text, re.I)
        if m:
            candidate = _clean_name(m.group(1))
            # Special case: "it's Nithin, Nithin Jain" -> take the fuller name
            # after the comma.
            tail = re.search(r",\s*([A-Za-z][A-Za-z .'-]+)$", text.strip())
            if tail:
                return _clean_name(tail.group(1))
            return candidate
        # 3) Bare name: they just typed "Nithin Jain" with no cue words.
        #    We only accept it if it's 2-5 words, all letters, and contains none
        #    of the filler/stopwords - so "tell me a joke" is NOT taken as a
        #    name. (This is a rough rule; the LLM handles the trickier cases.)
        stripped = text.strip().strip(".")
        tokens = stripped.split()
        if (
            2 <= len(tokens) <= 5
            and re.fullmatch(r"[A-Za-z][A-Za-z .'-]+", stripped)
            and not re.search(r"\b(dob|born|aadhaar|pin|account|acc|cvv|card)\b", stripped, re.I)
            and not any(tok.lower() in _NAME_STOPWORDS for tok in tokens)
        ):
            return _clean_name(stripped)
        return None

    # --- amount ------------------------------------------------------------ #
    # Figures out how much the user wants to pay.
    def _amount(self, text: str, result: ExtractionResult) -> None:
        low = text.lower()
        # "pay the full amount" / "clear it all" -> flag it; the code will use
        # the exact balance. (We don't need a number in this case.)
        if re.search(r"\b(full|entire|whole|clear it all|clear the full|all of it|total)\b", low):
            result.pay_full_balance = True
        # Explicit numeric amount, e.g. "500", "1000", "1,000.00", "540.5".
        # Match a full run of digits (optionally comma-grouped) with an optional
        # decimal part. Ordering matters: we capture the WHOLE number, not just
        # its first 1-3 digits, so "1000" stays 1000 (not 100).
        compact = text.replace(" ", "")
        m = re.search(r"(?<![\d.])(\d[\d,]*(?:\.\d+)?)", compact)
        if m:
            cleaned = m.group(1).replace(",", "")
            try:
                result.amount = float(cleaned)
            except ValueError:
                pass
        # word amounts: "a thousand rupees", "two hundred"
        if result.amount is None:
            words = self._word_amount(low)
            if words is not None:
                result.amount = words

    @staticmethod
    def _word_amount(low: str) -> Optional[float]:
        # Read amounts written in words. Walks the words left to right, adding
        # single numbers and applying multipliers.
        # Examples: "a thousand" -> 1000, "two hundred" -> 200,
        #           "two thousand five hundred" -> 2500.
        total = 0
        current = 0
        found = False
        for tok in re.findall(r"[a-z]+", low):
            if tok in _NUMBER_WORDS:
                current += _NUMBER_WORDS[tok]
                found = True
            elif tok in _MULTIPLIERS:
                mult = _MULTIPLIERS[tok]
                current = (current or 1) * mult
                total += current
                current = 0
                found = True
        total += current
        return float(total) if found and total > 0 else None

    # --- card -------------------------------------------------------------- #
    # Pulls the four card fields (number, expiry, cvv, name) from the text.
    def _card(self, text: str, result: ExtractionResult) -> None:
        # Card number: a run of 12-19 digits, spaces/dashes allowed.
        # "4532 0151 1283 0366" -> "4532015112830366".
        m = re.search(r"(?:\d[\s-]*){12,19}", text)
        if m:
            digits = re.sub(r"\D", "", m.group(0))
            if 12 <= len(digits) <= 19:
                result.card_number = digits

        # Expiry: MM/YY, MM/YYYY, "December 2027", "12-2027".
        month, year = self._expiry(text)
        if month:
            result.expiry_month = month
        if year:
            result.expiry_year = year

        # CVV: labelled, or spoken "one two three".
        cvv = self._cvv(text)
        if cvv:
            result.cvv = cvv

        # Cardholder name if explicitly stated. Handles "name on card X",
        # "cardholder name is X", and a trailing "name X" in card context.
        m = re.search(
            r"(?:name on (?:the )?card|cardholder(?:'s)?(?: name)?|name(?: is)?)\s*(?:is|:)?\s*([A-Za-z][A-Za-z .'-]+)",
            text,
            re.I,
        )
        if m:
            result.cardholder_name = _clean_name(m.group(1))

    @staticmethod
    def _expiry(text: str):
        # Read a card expiry as (month, year).
        # Handles "December 2027", "12/27", "12-2027". 2-digit years -> 20xx.
        months = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        # textual month + year
        mt = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*(\d{2,4})\b", text, re.I)
        if mt:
            month = months[mt.group(1)[:3].lower()]
            year = int(mt.group(2))
            year += 2000 if year < 100 else 0
            return month, year
        # MM/YY or MM/YYYY (guard against being a date-of-birth with 3 parts)
        mm = re.search(r"\b(0?[1-9]|1[0-2])\s*[/-]\s*(\d{2,4})\b", text)
        if mm:
            month = int(mm.group(1))
            year = int(mm.group(2))
            year += 2000 if year < 100 else 0
            return month, year
        return None, None

    @staticmethod
    def _cvv(text: str) -> Optional[str]:
        # Read the CVV. Needs the word "cvv" nearby (so we don't grab a random
        # 3-digit number). Handles both "cvv 123" and "cvv one two three".
        m = re.search(r"\bcvv\b[^0-9a-z]*([0-9]{3,4})", text, re.I)
        if m:
            return m.group(1)
        m2 = re.search(r"\bcvv\b[^0-9]*((?:(?:zero|one|two|three|four|five|six|seven|eight|nine)\s*){3,4})", text, re.I)
        if m2:
            digits = _words_to_digits(m2.group(1))
            if 3 <= len(digits) <= 4:
                return digits
        return None


def _clean_name(raw: str) -> str:
    """
    Tidy up a captured name: trim spaces/punctuation and cut off any trailing
    filler the regex may have grabbed.
    Example: "Nithin Jain and my dob is" -> "Nithin Jain".
    """
    name = raw.strip().strip(".,")
    # Drop trailing helper clauses if the regex over-captured.
    name = re.split(r"\b(and|my|dob|born|aadhaar|pin|account)\b", name, flags=re.I)[0]
    return re.sub(r"\s+", " ", name).strip()
