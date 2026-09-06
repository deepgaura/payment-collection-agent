"""Tests for the rule-based extractor against the assignment's messy inputs.

These run the deterministic extractor (the LLM path is exercised separately and
is not required for CI). Every example table from the assignment is covered.
"""

import pytest

from payment_agent.extractors.base import Expecting
from payment_agent.extractors.rule_based import RuleBasedExtractor

EX = RuleBasedExtractor()


# --- Account ID ------------------------------------------------------------ #
@pytest.mark.parametrize("text", [
    "yeah my account number is ACC1001 I think",
    "it's ACC 1001",
    "account id: acc1001",
])
def test_account_id_variants(text):
    assert EX.extract(text, Expecting.ACCOUNT).account_id == "ACC1001"


# --- Full name ------------------------------------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("my name is Nithin Jain", "Nithin Jain"),
    ("Nithin Jain", "Nithin Jain"),
    ("you can call me Raja but my full name is Rajarajeswari Balasubramaniam",
     "Rajarajeswari Balasubramaniam"),
])
def test_name_variants(text, expected):
    assert EX.extract(text, Expecting.IDENTITY).full_name == expected


# --- Date of birth (raw text captured, parsed downstream) ------------------ #
@pytest.mark.parametrize("text", [
    "I was born on 14th May 1990",
    "DOB is May 14, 90",
    "14-05-1990",
])
def test_dob_text_captured(text):
    assert EX.extract(text, Expecting.IDENTITY).dob_text is not None


# --- Aadhaar / pincode ----------------------------------------------------- #
def test_aadhaar_last4_labelled():
    assert EX.extract("last four of my Aadhaar is 4321", Expecting.IDENTITY).aadhaar_last4 == "4321"


def test_pincode_spaced_digits():
    assert EX.extract("pincode? it's 4 0 0 0 0 1", Expecting.IDENTITY).pincode == "400001"


def test_aadhaar_ends_with():
    r = EX.extract("Aadhaar ends with 9876, shall I give pincode instead?", Expecting.IDENTITY)
    assert r.aadhaar_last4 == "9876"


def test_account_token_digits_not_mistaken_for_aadhaar():
    """Digits inside an account token (e.g. "ACC-1001") must NOT be captured as
    an Aadhaar/pincode - otherwise restating an account id would pollute the
    identity claim and waste a verification attempt."""
    r = EX.extract("Sorry, I mean my account was ACC-1001", Expecting.IDENTITY)
    assert r.aadhaar_last4 is None
    assert r.pincode is None
    # A real labelled factor alongside an account token is still captured.
    r2 = EX.extract("my aadhaar is 4321 and acc ACC1001", Expecting.IDENTITY)
    assert r2.aadhaar_last4 == "4321"


# --- Amount ---------------------------------------------------------------- #
def test_amount_words():
    assert EX.extract("I want to pay a thousand rupees", Expecting.AMOUNT).amount == 1000.0


def test_amount_full_balance_flag():
    assert EX.extract("just clear the full amount", Expecting.AMOUNT).pay_full_balance


def test_amount_partial_number():
    assert EX.extract("can I do 500 for now?", Expecting.AMOUNT).amount == 500.0


@pytest.mark.parametrize("text,expected", [
    ("pay 1000", 1000.0),      # regression: plain 4-digit must not become 100
    ("1000", 1000.0),
    ("pay 1,000", 1000.0),
    ("pay 3200.50", 3200.50),
    ("pay 12000", 12000.0),
])
def test_amount_digit_forms(text, expected):
    assert EX.extract(text, Expecting.AMOUNT).amount == expected


# --- Card details ---------------------------------------------------------- #
def test_card_number_grouped():
    r = EX.extract("the card number is 4532 0151 1283 0366", Expecting.CARD)
    assert r.card_number == "4532015112830366"


@pytest.mark.parametrize("text,month,year", [
    ("expires December 2027", 12, 2027),
    ("12/27", 12, 2027),
])
def test_expiry_variants(text, month, year):
    r = EX.extract(text, Expecting.CARD)
    assert (r.expiry_month, r.expiry_year) == (month, year)


def test_cvv_spoken():
    assert EX.extract("CVV is one two three", Expecting.CARD).cvv == "123"


# --- Disambiguation by context --------------------------------------------- #
def test_bare_number_is_amount_when_expecting_amount():
    r = EX.extract("500", Expecting.AMOUNT)
    assert r.amount == 500.0


def test_quit_intent():
    assert EX.extract("actually cancel this", Expecting.ACCOUNT).wants_to_quit


def test_unclear_secondary_flag_on_wrong_length_number():
    # A 5-digit number is neither Aadhaar (4) nor pincode (6) -> flag as unclear.
    r = EX.extract("43215", Expecting.IDENTITY)
    assert r.aadhaar_last4 is None and r.pincode is None
    assert r.unclear_secondary


def test_valid_factor_not_flagged_unclear():
    assert not EX.extract("4321", Expecting.IDENTITY).unclear_secondary
    assert not EX.extract("400001", Expecting.IDENTITY).unclear_secondary


# --- LLM extractor: structured-output path (offline, fake client) ---------- #
# These prove the structured-output wiring WITHOUT hitting a real LLM: we inject
# a fake client that returns a dict, exactly like complete_json would.
from payment_agent.extractors.llm import LLMExtractor
from payment_agent.extractors.schema import EXTRACTION_SCHEMA
from payment_agent.config import Config

# A complete, schema-valid extraction object (all 14 keys, value-or-null).
_VALID_LLM_OBJECT = {
    "account_id": "ACC1001", "full_name": None, "dob_text": None,
    "aadhaar_last4": None, "pincode": None, "amount": None,
    "pay_full_balance": False, "cardholder_name": None, "card_number": None,
    "cvv": None, "expiry_month": None, "expiry_year": None,
    "wants_to_quit": False, "confirm": None,
}


class _FakeClient:
    """Stands in for LLMClient. Records the schema it was given and returns a
    canned dict (or raises) so we can test the extractor deterministically."""

    def __init__(self, to_return=None, raise_exc=None):
        self._to_return = to_return
        self._raise = raise_exc
        self.received_schema = "unset"

    def complete_json(self, system, user, schema=None):
        self.received_schema = schema
        if self._raise is not None:
            raise self._raise
        return dict(self._to_return)


def test_llm_extractor_passes_schema_to_client():
    """The extractor must hand the JSON Schema to the client (provider-level
    enforcement), not just a bare prompt."""
    fake = _FakeClient(to_return=_VALID_LLM_OBJECT)
    ex = LLMExtractor(Config(), client=fake)
    result = ex.extract("my account is acc 1001", Expecting.ACCOUNT)
    assert fake.received_schema is EXTRACTION_SCHEMA      # schema was forwarded
    assert result.account_id == "ACC1001"
    assert result.source == "llm"


def test_llm_extractor_rejects_schema_violation_and_falls_back():
    """If the model returns a shape that violates the schema (here: extra key +
    wrong type), the validation gate rejects it and we fall back to regex, with
    a 'schema_invalid' note so it's observable in metrics."""
    bad = {"account_id": 12345, "surprise_field": "nope"}  # wrong type + extra key
    fake = _FakeClient(to_return=bad)
    ex = LLMExtractor(Config(), client=fake)
    result = ex.extract("my account is acc 1001", Expecting.ACCOUNT)
    # Fell back to the deterministic regex extractor...
    assert result.source == "rule_based"
    assert "schema_invalid" in result.notes
    # ...which still correctly reads the account id from the same text.
    assert result.account_id == "ACC1001"


def test_llm_extractor_network_failure_falls_back_as_llm_failed():
    """A non-schema error (e.g. network) tags 'llm_failed', not 'schema_invalid'."""
    fake = _FakeClient(raise_exc=ConnectionError("boom"))
    ex = LLMExtractor(Config(), client=fake)
    result = ex.extract("acc 1001", Expecting.ACCOUNT)
    assert result.source == "rule_based"
    assert "llm_failed" in result.notes
    assert "schema_invalid" not in result.notes


def test_extraction_schema_is_self_consistent():
    """Every 'required' key must be defined in properties, and extra keys are
    forbidden - guards against future edits drifting the schema."""
    props = set(EXTRACTION_SCHEMA["properties"])
    required = set(EXTRACTION_SCHEMA["required"])
    assert required == props                        # all keys required, none missing
    assert EXTRACTION_SCHEMA["additionalProperties"] is False
