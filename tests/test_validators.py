"""Unit tests for the deterministic validators."""

import datetime as dt

import pytest

from payment_agent.validators import (
    luhn_ok,
    parse_date_strict,
    validate_amount,
    validate_card_number,
    validate_cvv,
    validate_expiry,
)


# --- Luhn / card number ---------------------------------------------------- #
def test_luhn_valid_card():
    assert luhn_ok("4532015112830366")


def test_luhn_invalid_card():
    assert not luhn_ok("1234567890123456")


def test_validate_card_number_normalises_spaces():
    r = validate_card_number("4532 0151 1283 0366")
    assert r.ok and r.value == "4532015112830366"


def test_validate_card_number_rejects_masked():
    assert not validate_card_number("4532 **** **** 0366").ok


def test_validate_card_number_rejects_luhn_fail():
    assert not validate_card_number("1234567890123456").ok


# --- CVV ------------------------------------------------------------------- #
def test_cvv_three_digits_ok():
    assert validate_cvv("123", "4532015112830366").ok


def test_cvv_wrong_length_rejected():
    assert not validate_cvv("12", "4532015112830366").ok


def test_cvv_amex_requires_four():
    # Amex BIN starts 34/37.
    assert validate_cvv("1234", "371449635398431").ok
    assert not validate_cvv("123", "371449635398431").ok


# --- Expiry ---------------------------------------------------------------- #
def test_expiry_future_ok():
    assert validate_expiry(12, 2030, today=dt.date(2026, 1, 1)).ok


def test_expiry_two_digit_year_normalised():
    r = validate_expiry(12, 27, today=dt.date(2026, 1, 1))
    assert r.ok and r.value == (12, 2027)


def test_expiry_past_rejected():
    assert not validate_expiry(1, 2020, today=dt.date(2026, 1, 1)).ok


def test_expiry_valid_through_end_of_month():
    # Card expiring this month is still valid until month-end.
    assert validate_expiry(1, 2026, today=dt.date(2026, 1, 15)).ok


def test_expiry_bad_month_rejected():
    assert not validate_expiry(13, 2030).ok


# --- Amount ---------------------------------------------------------------- #
def test_amount_ok_partial():
    r = validate_amount(500.0, 1250.75)
    assert r.ok and r.value == 500.0


def test_amount_full_balance_ok():
    assert validate_amount(1250.75, 1250.75).ok


def test_amount_zero_rejected():
    assert not validate_amount(0, 1000).ok


def test_amount_negative_rejected():
    assert not validate_amount(-5, 1000).ok


def test_amount_too_many_decimals_rejected():
    assert not validate_amount(10.999, 1000).ok


def test_amount_exceeds_balance_rejected():
    assert not validate_amount(2000, 1250.75).ok


# --- Strict date parsing (incl. leap year) --------------------------------- #
@pytest.mark.parametrize("text,expected", [
    ("1990-05-14", "1990-05-14"),
    ("I was born on 14th May 1990", "1990-05-14"),
    ("DOB is May 14, 90", "1990-05-14"),
    ("14-05-1990", "1990-05-14"),
    ("29th February 1988", "1988-02-29"),   # leap year: valid
])
def test_parse_date_strict_valid(text, expected):
    r = parse_date_strict(text)
    assert r.ok, f"expected to parse {text!r}"
    assert r.value == expected


@pytest.mark.parametrize("text", [
    "30th February 1990",   # Feb 30 never exists
    "1989-02-29",           # 1989 not a leap year
    "not a date at all",
])
def test_parse_date_strict_invalid(text):
    assert not parse_date_strict(text).ok
