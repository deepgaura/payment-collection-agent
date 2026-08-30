"""Tests for strict identity verification."""

from payment_agent.state import Account, IdentityClaim
from payment_agent.verifier import SecondaryFactor, verify

ACCT = Account(
    account_id="ACC1001", full_name="Nithin Jain", dob="1990-05-14",
    aadhaar_last4="4321", pincode="400001", balance=1250.75,
)


def _claim(**kw):
    return IdentityClaim(**kw)


def test_name_plus_dob_verifies():
    out = verify(_claim(full_name="Nithin Jain", dob="1990-05-14"), ACCT)
    assert out.verified and out.factor_matched == SecondaryFactor.DOB


def test_name_plus_aadhaar_verifies():
    out = verify(_claim(full_name="Nithin Jain", aadhaar_last4="4321"), ACCT)
    assert out.verified and out.factor_matched == SecondaryFactor.AADHAAR


def test_name_plus_pincode_verifies():
    out = verify(_claim(full_name="Nithin Jain", pincode="400001"), ACCT)
    assert out.verified and out.factor_matched == SecondaryFactor.PINCODE


def test_wrong_name_fails_even_with_correct_factor():
    out = verify(_claim(full_name="Nithin Kumar", dob="1990-05-14"), ACCT)
    assert not out.verified and not out.name_matches


def test_name_matches_but_no_secondary_factor_fails():
    out = verify(_claim(full_name="Nithin Jain", dob="1999-01-01"), ACCT)
    assert not out.verified and out.name_matches


def test_name_is_case_sensitive_strict():
    # Hard rule: no case-insensitive matching.
    out = verify(_claim(full_name="nithin jain", dob="1990-05-14"), ACCT)
    assert not out.verified


def test_name_whitespace_collapse_allowed():
    # The single documented normalisation: internal whitespace is collapsed.
    out = verify(_claim(full_name="Nithin   Jain", dob="1990-05-14"), ACCT)
    assert out.verified


def test_aadhaar_ignores_spaces():
    # Numeric factors compare by value; separators are formatting (standard in
    # real KYC systems), so "4 3 2 1" == "4321". This is NOT fuzzy matching.
    out = verify(_claim(full_name="Nithin Jain", aadhaar_last4="4 3 2 1"), ACCT)
    assert out.verified


def test_aadhaar_ignores_dashes():
    out = verify(_claim(full_name="Nithin Jain", aadhaar_last4="43-21"), ACCT)
    assert out.verified


def test_aadhaar_with_letters_rejected():
    # Defence in depth: a factor containing junk is not salvaged into digits.
    out = verify(_claim(full_name="Nithin Jain", aadhaar_last4="4321abc"), ACCT)
    assert not out.verified


def test_no_claim_fails():
    assert not verify(_claim(), ACCT).verified
