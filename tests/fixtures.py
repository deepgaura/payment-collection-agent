"""Test fixtures: sample accounts and a deterministic mock API client.

Using a mock API keeps the suite fast, offline, and reproducible. The mock
mirrors the real server's observed behaviour, including the quirk that bad
expiry / CVV come back as HTTP 400 `invalid_args` rather than the granular
codes in the docs.
"""

from __future__ import annotations

from payment_agent.tools.payment_api import ApiResult
from payment_agent.validators import is_amex, luhn_ok

# The four documented sample accounts.
ACCOUNTS = {
    "ACC1001": {
        "account_id": "ACC1001", "full_name": "Nithin Jain", "dob": "1990-05-14",
        "aadhaar_last4": "4321", "pincode": "400001", "balance": 1250.75,
    },
    "ACC1002": {
        "account_id": "ACC1002", "full_name": "Rajarajeswari Balasubramaniam",
        "dob": "1985-11-23", "aadhaar_last4": "9876", "pincode": "400002",
        "balance": 540.00,
    },
    "ACC1003": {
        "account_id": "ACC1003", "full_name": "Priya Agarwal", "dob": "1992-08-10",
        "aadhaar_last4": "2468", "pincode": "400003", "balance": 0.00,
    },
    "ACC1004": {
        "account_id": "ACC1004", "full_name": "Rahul Mehta", "dob": "1988-02-29",
        "aadhaar_last4": "1357", "pincode": "400004", "balance": 3200.50,
    },
}


class MockApiClient:
    """Drop-in replacement for PaymentApiClient with the same public methods."""

    def __init__(self):
        self.lookup_calls: list[str] = []
        self.payment_calls: list[dict] = []
        self._txn_counter = 0

    def lookup_account(self, account_id: str) -> ApiResult:
        self.lookup_calls.append(account_id)
        acct = ACCOUNTS.get(account_id)
        if acct is None:
            return ApiResult(
                ok=False, status_code=404, error_code="account_not_found",
                message="No account found with the provided account_id.",
            )
        return ApiResult(ok=True, status_code=200, data=dict(acct))

    def process_payment(
        self, *, account_id, amount, cardholder_name, card_number, cvv,
        expiry_month, expiry_year,
    ) -> ApiResult:
        self.payment_calls.append({
            "account_id": account_id, "amount": amount,
            "card_number": card_number, "cvv": cvv,
            "expiry_month": expiry_month, "expiry_year": expiry_year,
        })
        acct = ACCOUNTS.get(account_id)
        if acct is None:
            return ApiResult(ok=False, status_code=404, error_code="account_not_found")

        # Mirror the server's validation order/behaviour.
        if not luhn_ok(card_number) or not (12 <= len(card_number) <= 19):
            return ApiResult(ok=False, status_code=422, error_code="invalid_card")
        expected_cvv = 4 if is_amex(card_number) else 3
        if len(cvv) != expected_cvv:
            return ApiResult(ok=False, status_code=400, error_code="invalid_args")
        if not (1 <= expiry_month <= 12) or expiry_year < 2025:
            return ApiResult(ok=False, status_code=400, error_code="invalid_args")
        if amount <= 0 or round(amount, 2) != amount:
            return ApiResult(ok=False, status_code=422, error_code="invalid_amount")
        if round(amount, 2) > round(acct["balance"], 2):
            return ApiResult(ok=False, status_code=422, error_code="insufficient_balance")

        self._txn_counter += 1
        return ApiResult(
            ok=True, status_code=200,
            data={"success": True, "transaction_id": f"txn_test_{self._txn_counter}"},
        )


class NetworkFailApiClient:
    """Simulates persistent network failure for both endpoints."""

    def lookup_account(self, account_id: str) -> ApiResult:
        return ApiResult(ok=False, network_error=True, message="connection refused")

    def process_payment(self, **kwargs) -> ApiResult:
        return ApiResult(ok=False, network_error=True, message="connection refused")
