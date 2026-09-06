"""
WHAT THIS FILE IS (in one line):
    The SINGLE SOURCE OF TRUTH for the shape of what the LLM extracts.

WHY THIS FILE EXISTS:
    Before, the "shape" lived in two places that could drift apart:
      - a description written in the extractor's prompt, and
      - the Python code that read the reply.
    That's a classic bug source (change one, forget the other). Here we define
    the shape ONCE as a JSON Schema, and everyone else points at it:
      - llm_client.py    -> hands this schema to the provider so the MODEL is
                            forced to return exactly this shape (OpenAI's
                            json_schema mode / Claude's tool input_schema).
      - llm.py           -> validates the reply against this schema as a final
                            gate before trusting it.

    So "structured output" is enforced in three places at once: the model layer,
    a schema-validation gate, and (still) our defensive type coercion. Belt and
    suspenders - we never fully trust the model even when it promises a schema.

THE 14 FIELDS:
    These mirror ExtractionResult in base.py (the typed object the app uses).
    Each is either a value or null (we require every key to be present, and
    allow null when the info isn't in the message - that's how the model says
    "not stated" instead of leaving it out).
"""

from __future__ import annotations

# The name we give the "function/tool" when using Claude's tool-based structured
# output. (OpenAI uses it as the json_schema name.) Kept as a constant so the
# client and any tests refer to the same string.
EXTRACTION_TOOL_NAME = "extract_payment_fields"

# One human-readable line describing the job, reused by both providers.
EXTRACTION_TOOL_DESCRIPTION = (
    "Extract explicitly-present payment/identity fields from ONE user message. "
    "Use null for anything not clearly stated. Never invent or repair values."
)

# ---------------------------------------------------------------------------
# THE SCHEMA (JSON Schema, draft 2020-12 compatible subset).
#
# Notes on the choices:
#   - Every field is `["<type>", "null"]` so the model can say "not present"
#     explicitly, which is safer than an optional/absent key.
#   - `additionalProperties: false` forbids the model from inventing extra keys.
#   - `required` lists ALL keys: with OpenAI strict mode the model must emit
#     every key (value or null), which removes "did it forget this field?"
#     ambiguity.
#   - We keep types loose enough to be robust (e.g. aadhaar/pincode/cvv as
#     strings of digits) and re-tighten/validate the exact digit-counts later
#     in deterministic code, where a wrong value is a re-ask, not a crash.
# ---------------------------------------------------------------------------
EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "account_id": {
            "type": ["string", "null"],
            "description": (
                'Only if the text contains "ACC"+digits (case/space aside), '
                'e.g. "acc 1001"->"ACC1001". Do NOT invent or repair: "AC1001", '
                '"A1001", or a bare number are NOT account ids -> null.'
            ),
        },
        "full_name": {
            "type": ["string", "null"],
            "description": "The person's FULL name if stated.",
        },
        "dob_text": {
            "type": ["string", "null"],
            "description": "The RAW date phrase exactly as written; do NOT reformat.",
        },
        "aadhaar_last4": {
            "type": ["string", "null"],
            "description": "Exactly 4 digits if the user gives Aadhaar last 4.",
        },
        "pincode": {
            "type": ["string", "null"],
            "description": "Exactly 6 digits if the user gives a pincode.",
        },
        "amount": {
            "type": ["number", "null"],
            "description": 'Numeric amount, e.g. "a thousand rupees" -> 1000.',
        },
        "pay_full_balance": {
            "type": "boolean",
            "description": "true if user asks to pay the full/entire/whole outstanding amount.",
        },
        "cardholder_name": {
            "type": ["string", "null"],
            "description": "Name on the card if stated.",
        },
        "card_number": {
            "type": ["string", "null"],
            "description": "Digits only; strip spaces/dashes.",
        },
        "cvv": {
            "type": ["string", "null"],
            "description": 'Digits only, e.g. "one two three" -> "123".',
        },
        "expiry_month": {
            "type": ["integer", "null"],
            "description": "1-12.",
        },
        "expiry_year": {
            "type": ["integer", "null"],
            "description": '4-digit year, e.g. "27" -> 2027.',
        },
        "wants_to_quit": {
            "type": "boolean",
            "description": "true if the user wants to cancel/stop.",
        },
        "confirm": {
            "type": ["boolean", "null"],
            "description": (
                "At a yes/no confirmation: true=go ahead, false=no/change, "
                "null=neither."
            ),
        },
    },
    # ALL keys required (value-or-null). This is what makes the output truly
    # predictable field-by-field.
    "required": [
        "account_id", "full_name", "dob_text", "aadhaar_last4", "pincode",
        "amount", "pay_full_balance", "cardholder_name", "card_number", "cvv",
        "expiry_month", "expiry_year", "wants_to_quit", "confirm",
    ],
}
