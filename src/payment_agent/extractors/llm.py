"""LLM-backed extractor (primary NLU layer).

Responsibility (and ONLY this): read the user's raw message plus a hint about
what the agent currently expects, and return structured candidate fields as
JSON. It never verifies, validates, decides flow, or sees stored account data.

Prompt-engineering choices, and why:
- Single, tightly-scoped task ("extract fields to JSON"). Narrow tasks are
  where LLMs are most reliable; we do not ask it to reason about the flow.
- Strict JSON schema via response_format=json_object + an explicit schema in
  the system prompt, so output is machine-parseable every turn.
- Temperature 0 for maximum determinism.
- "Only extract what is explicitly present; use null otherwise" - this
  suppresses hallucination, which is critical when the fields feed identity
  verification and payment.
- Dates are returned as the user's raw phrase (dob_text); we parse/normalise
  them with deterministic code, not the model, so leap-year / format edge
  cases are handled predictably.
- Defence in depth: the model output is merged with the deterministic
  rule-based extractor and then fully re-validated downstream. A wrong or
  adversarial extraction cannot bypass verification or validation.
- Robustness: any error (timeout, bad JSON, missing key) falls back to the
  rule-based extractor. The agent never breaks because the LLM misbehaved.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from ..config import Config
from .base import Expecting, ExtractionResult
from .rule_based import RuleBasedExtractor

logger = logging.getLogger("payment_agent.llm")

SYSTEM_PROMPT = """\
You are a precise information-extraction component inside a payment-collection \
agent. You do NOT talk to the user, make decisions, or verify anything. Your \
ONLY job is to read one user message and extract explicitly-present fields into \
a strict JSON object.

Return a JSON object with EXACTLY these keys (use null when a field is not \
clearly present in THIS message):

{
  "account_id": string|null,        // normalise to "ACC" + digits, e.g. "acc 1001" -> "ACC1001"
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
  "wants_to_quit": boolean          // true if the user wants to cancel/stop
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
    "expiry_month", "expiry_year", "wants_to_quit",
}


class LLMExtractor:
    """Primary extractor. Delegates to `fallback` on any failure."""

    source = "llm"

    def __init__(self, config: Config, fallback: Optional[RuleBasedExtractor] = None):
        self._config = config
        self._fallback = fallback or RuleBasedExtractor()
        # Import lazily so the package works without the openai dependency.
        from openai import OpenAI

        self._client = OpenAI(api_key=config.llm_api_key, timeout=config.llm_timeout_seconds)

    def extract(self, text: str, expecting: Expecting) -> ExtractionResult:
        rule_based = self._fallback.extract(text, expecting)
        try:
            llm_result = self._call_llm(text, expecting)
        except Exception as exc:  # timeout, network, bad JSON, SDK error...
            logger.warning("LLM extraction failed (%s); using rule-based only.", exc.__class__.__name__)
            rule_based.notes.append("llm_failed")
            return rule_based

        # LLM is primary; deterministic regex fills any gaps it missed.
        llm_result.merge_missing_from(rule_based)
        return llm_result

    def _call_llm(self, text: str, expecting: Expecting) -> ExtractionResult:
        user_prompt = (
            f'expecting: "{expecting.value}"\n'
            f'user_message: {json.dumps(text)}\n'
            f"Extract the fields as specified and return ONLY the JSON object."
        )
        resp = self._client.chat.completions.create(
            model=self._config.llm_model,
            temperature=self._config.llm_temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        return self._to_result(data)

    def _to_result(self, data: dict) -> ExtractionResult:
        """Coerce raw LLM JSON into a typed, sanitised ExtractionResult.

        We defensively normalise types and ignore unexpected keys, so a
        slightly-off model response can never corrupt downstream logic.
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
        return result


# --- defensive coercion helpers -------------------------------------------- #
def _as_str(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _as_digits(v) -> Optional[str]:
    if v is None:
        return None
    d = re.sub(r"\D", "", str(v))
    return d or None


def _as_int(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
