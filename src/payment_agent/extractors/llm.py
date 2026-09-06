"""
WHAT THIS FILE IS (in one line):
    The smart translator: it asks an LLM to read the user's messy sentence and
    return clean fields as JSON.

HOW IT WORKS (3 steps):
    1. Send the user's message + a rulebook (SYSTEM_PROMPT) to the LLM.
    2. The LLM replies with JSON like {"account_id": "ACC1001", ...}.
    3. We clean/sanity-check that JSON and return it as an ExtractionResult.

SAFETY / KEY IDEAS:
    - The LLM ONLY extracts data. It never verifies, never decides, never sees
      the stored account. So it can't bypass any security check.
    - "Only extract what's actually there; otherwise null" -> stops the LLM from
      making things up (very important for identity/card fields).
    - If the LLM fails for ANY reason (timeout, bad JSON), we fall back to the
      rule-based (regex) translator. The agent never breaks.
    - The LLM's answer is later merged with the regex answer AND re-validated,
      so a wrong guess can't slip through.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from ..config import Config
from .base import Expecting, ExtractionResult
from .rule_based import RuleBasedExtractor
from .schema import EXTRACTION_SCHEMA

logger = logging.getLogger("payment_agent.llm")


def _validate_against_schema(data: dict) -> None:
    """
    Final gate: check the model's reply really matches our schema.

    Provider schema-enforcement is great but not universal (older libs, plain
    JSON fallback). So we ALSO validate here. If jsonschema isn't installed we
    skip silently - the strict type-coercion in _to_result is still a safety
    net. A schema violation raises, which the caller turns into a regex
    fallback (never a crash).
    """
    try:
        import jsonschema
    except Exception:
        return  # library not present -> rely on coercion instead
    jsonschema.validate(instance=data, schema=EXTRACTION_SCHEMA)


def _is_schema_error(exc: Exception) -> bool:
    """True if this exception came from schema validation (vs a network/LLM
    failure). Used only to tag the fallback note for metrics/debugging."""
    return exc.__class__.__name__ in ("ValidationError", "SchemaError")

# ---------------------------------------------------------------------------
# THE RULEBOOK WE SEND THE LLM (the "system prompt").
# This is the actual text sent to the model. It tells it: "return ONLY this
# exact JSON shape, fill in what's clearly present, use null otherwise, and
# never invent anything." Getting this prompt right is the core of making the
# LLM reliable - so we're very explicit about every field.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a precise information-extraction component inside a payment-collection \
agent. You do NOT talk to the user, make decisions, or verify anything. Your \
ONLY job is to read one user message and extract explicitly-present fields into \
a strict JSON object.

Return a JSON object with EXACTLY these keys (use null when a field is not \
clearly present in THIS message):

{
  "account_id": string|null,        // ONLY if the user's text contains "ACC"+digits (case/space aside), e.g. "acc 1001"->"ACC1001". Do NOT invent or repair: "AC1001", "A1001", or a bare number are NOT account ids -> null.
  "full_name": string|null,         // the person's FULL name if stated; prefer an explicit "full name is X"
  "dob_text": string|null,          // the raw date phrase EXACTLY as the user wrote it; do NOT reformat
  "aadhaar_last4": string|null,     // exactly 4 digits if the user gives Aadhaar last 4
  "pincode": string|null,           // exactly 6 digits if the user gives a pincode
  "amount": number|null,            // numeric amount, e.g. "a thousand rupees" -> 1000
  "pay_full_balance": boolean,      // true if user asks to pay the full/entire/whole outstanding amount
  "cardholder_name": string|null,   // name on the card if stated
  "card_number": string|null,       // digits only, strip spaces/dashes
  "cvv": string|null,               // digits only, e.g. "one two three" -> "123"
  "expiry_month": integer|null,     // 1-12
  "expiry_year": integer|null,      // 4-digit year, e.g. "27" -> 2027
  "wants_to_quit": boolean,         // true if the user wants to cancel/stop
  "confirm": true|false|null        // at a yes/no confirmation: true=go ahead, false=no/change, null=neither
}

Rules:
- Extract ONLY what is explicitly present. Never guess or invent values. When \
unsure, use null. Hallucinated identity or card data is a serious error.
- Do NOT reformat dates. Put the user's exact wording in dob_text.
- Separate Aadhaar-last-4 (4 digits) from pincode (6 digits) using the user's \
own wording; if ambiguous, prefer the labelled one and leave the other null.
- "expecting" tells you what the agent is currently collecting; use it to \
disambiguate bare numbers, but still capture clearly-labelled out-of-context \
fields (e.g. a stated account id).
- Output ONLY the JSON object. No prose, no markdown.
"""

_ALLOWED_KEYS = {
    "account_id", "full_name", "dob_text", "aadhaar_last4", "pincode",
    "amount", "pay_full_balance", "cardholder_name", "card_number", "cvv",
    "expiry_month", "expiry_year", "wants_to_quit", "confirm",
}

class LLMExtractor:
    """
    The smart (LLM) translator.

    Holds two things:
      - self._client   : the connection to the LLM (Claude/OpenAI)
      - self._fallback : the rule-based (regex) translator, used if the LLM fails
    """

    source = "llm"   # stamped onto results so we know the LLM produced them

    def __init__(self, config: Config, fallback: Optional[RuleBasedExtractor] = None, client=None):
        self._config = config
        # Backup translator (regex). Used whenever the LLM can't answer.
        self._fallback = fallback or RuleBasedExtractor()
        # The LLM connection (works for Claude or OpenAI). Built here unless a
        # test injects its own.
        if client is None:
            from ..llm_client import LLMClient

            client = LLMClient(config)
        self._client = client

    def extract(self, text: str, expecting: Expecting) -> ExtractionResult:
        # 1) Run the cheap, always-works regex translator first. We'll use it to
        #    fill any boxes the LLM leaves empty.
        rule_based = self._fallback.extract(text, expecting)

        # 2) Try the LLM.
        try:
            llm_result = self._call_llm(text, expecting)
        except Exception as exc:
            # If the LLM errored (timeout/network/bad JSON/schema violation),
            # DON'T crash - use the regex result and note why we fell back.
            note = "schema_invalid" if _is_schema_error(exc) else "llm_failed"
            logger.warning(
                "LLM extraction failed (%s: %s); using rule-based only.",
                note, exc.__class__.__name__,
            )
            rule_based.notes.append(note)
            return rule_based

        # 3) LLM leads; regex fills the gaps. Return the combined form.
        llm_result.merge_missing_from(rule_based)
        return llm_result

    def _call_llm(self, text: str, expecting: Expecting) -> ExtractionResult:
        # Build the "user message" part: what the user said + what we expect.
        # (The rulebook is the SYSTEM_PROMPT defined above.)
        user_prompt = (
            f'expecting: "{expecting.value}"\n'
            f'user_message: {json.dumps(text)}\n'
            f"Extract the fields as specified and return ONLY the JSON object."
        )
        # Ask the LLM for JSON, forcing the provider to match our schema where
        # supported. We pass the schema so structure is enforced at the source.
        data = self._client.complete_json(SYSTEM_PROMPT, user_prompt, schema=EXTRACTION_SCHEMA)
        # Final gate: validate the reply really matches the schema before we
        # trust it. (Raises on violation -> caller falls back to regex.)
        _validate_against_schema(data)
        return self._to_result(data)

    def _to_result(self, data: dict) -> ExtractionResult:
        """
        Turn the LLM's raw JSON into a clean ExtractionResult.

        We do NOT blindly trust the JSON. We keep only recognised keys and force
        each value into the right type, so a slightly-wrong reply can't corrupt
        the rest of the app.
        """
        clean = {k: v for k, v in data.items() if k in _ALLOWED_KEYS}
        result = ExtractionResult(source=self.source)

        result.account_id = _as_str(clean.get("account_id"))
        result.full_name = _as_str(clean.get("full_name"))
        result.dob_text = _as_str(clean.get("dob_text"))
        result.aadhaar_last4 = _as_digits(clean.get("aadhaar_last4"))
        result.pincode = _as_digits(clean.get("pincode"))
        result.amount = _as_float(clean.get("amount"))
        result.pay_full_balance = bool(clean.get("pay_full_balance", False))
        result.cardholder_name = _as_str(clean.get("cardholder_name"))
        result.card_number = _as_digits(clean.get("card_number"))
        result.cvv = _as_digits(clean.get("cvv"))
        result.expiry_month = _as_int(clean.get("expiry_month"))
        result.expiry_year = _as_int(clean.get("expiry_year"))
        result.wants_to_quit = bool(clean.get("wants_to_quit", False))
        c = clean.get("confirm", None)
        result.confirm = c if isinstance(c, bool) else None
        return result


# --- small "make sure it's the right type" helpers ------------------------- #
# The LLM's JSON values might be slightly off (a number as text, extra spaces,
# etc.). These helpers force each value into what we expect, returning None if
# it can't be made sense of - so bad data becomes "empty", never a crash.

def _as_str(v) -> Optional[str]:
    # Return a trimmed string, or None if empty/missing. e.g. "  Nithin " -> "Nithin"
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _as_digits(v) -> Optional[str]:
    # Keep only the digits. e.g. "4532 0151" -> "45320151", "abc" -> None
    if v is None:
        return None
    d = re.sub(r"\D", "", str(v))
    return d or None


def _as_int(v) -> Optional[int]:
    # Turn into a whole number, or None. e.g. "12" -> 12, "dec" -> None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v) -> Optional[float]:
    # Turn into a decimal number, or None. e.g. "500" -> 500.0, "lots" -> None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
