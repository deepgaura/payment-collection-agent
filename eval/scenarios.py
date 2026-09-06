"""
WHAT THIS FILE IS (in one line):
    The list of scripted test conversations + the little "check" functions that
    say what SHOULD happen at each step. run_eval.py plays these.

THE SHAPE OF A SCENARIO:
    A Scenario = a name, a category, and a list of Turns.
    A Turn     = one user message + a list of checks to run on the agent's reply.
    A check    = a tiny function that returns (passed?, description).
    So a scenario reads almost like a chat transcript with assertions attached.

WHAT "CORRECT" MEANS (checked here):
    - greeting     -> agent asks for the account id
    - lookup       -> the right account is fetched (a tool-call check)
    - verification -> passes ONLY when the strict rule holds; never leaks data
    - amount       -> matches what the user meant and respects the balance
    - payment      -> API called with the right data at the right time, and the
                      outcome is stated (txn id on success, reason on failure);
                      and NOT called when it shouldn't be (negative checks)
    - closure      -> the conversation ends in the right final state

    Checks look at both the agent's WORDS and its observable state
    (agent.step / is_verified / transaction_id) and the fake API's recorded
    calls - so we verify real behaviour, not just phrasing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

# A "check" is a function: given (agent's message, agent, api) it returns
# (did it pass?, a short description). Everything below builds these.
Check = Callable[[str, object, object], tuple[bool, str]]


# --- a fake API that always fails payment (for payment-failure scenarios) --- #
def _forced_error_client(error_code: str, status: int = 422):
    """
    Make a fake API whose process_payment ALWAYS returns the given error.
    Handy for testing server-side failures like "insufficient_balance" that our
    local checks would otherwise let through. (Lookup still works normally.)
    """
    from tests.fixtures import MockApiClient
    from payment_agent.tools.payment_api import ApiResult

    class _Forced(MockApiClient):
        def process_payment(self, **kwargs):
            super().process_payment(**kwargs)  # record the call for tool metrics
            return ApiResult(ok=False, status_code=status, error_code=error_code)

    return _Forced


@dataclass
class Turn:
    """One user message, plus the checks to run on the agent's reply to it."""
    user: str
    checks: list[Check] = field(default_factory=list)


@dataclass
class Scenario:
    """One full scripted conversation to test."""
    name: str
    category: str  # one of: happy | verification_failure | payment_failure | edge_case
    turns: list[Turn]
    # Checks to run AFTER all turns (e.g. "did we end in closed_success?").
    final_checks: list[Check] = field(default_factory=list)
    # Optional: bring a custom fake API (e.g. one that forces a server error).
    api_factory: object = None


# --- the little check-builder functions ------------------------------------ #
# Each returns a `check` function. We call them when building scenarios, e.g.
# msg_contains("card") makes a check that passes if the reply mentions "card".

def _tool_check(fn):
    """Mark a check as a 'tool-call correctness' check, so the report can count
    those separately (the assignment names this as a metric)."""
    fn.is_tool_check = True
    return fn


def msg_contains(*substrings: str) -> Check:
    # Passes if the agent's reply contains ALL of these words/phrases (case-insensitive).
    def check(message: str, agent, api) -> tuple[bool, str]:
        low = message.lower()
        ok = all(s.lower() in low for s in substrings)
        return ok, f"message contains {substrings!r}"
    return check


def msg_excludes(*substrings: str) -> Check:
    # Passes if the reply contains NONE of these (e.g. no leaked data).
    def check(message: str, agent, api) -> tuple[bool, str]:
        ok = all(s.lower() not in message.lower() for s in substrings)
        return ok, f"message excludes {substrings!r}"
    return check


def step_is(step_value: str) -> Check:
    # Passes if the agent is on this step (e.g. "closed_success").
    def check(message: str, agent, api) -> tuple[bool, str]:
        return agent.step.value == step_value, f"step == {step_value}"
    return check


def verified(expected: bool) -> Check:
    # Passes if the agent's verified flag matches what we expect (True/False).
    def check(message: str, agent, api) -> tuple[bool, str]:
        return agent.is_verified == expected, f"is_verified == {expected}"
    return check


def has_transaction() -> Check:
    # Passes if a real transaction id was produced (i.e. a payment went through).
    def check(message: str, agent, api) -> tuple[bool, str]:
        tid = agent.transaction_id
        return bool(tid and tid.startswith("txn_")), "transaction id present"
    return check


def payment_calls(n: int) -> Check:
    # Tool-call check: passes if the payment API was called exactly n times.
    @_tool_check
    def check(message: str, agent, api) -> tuple[bool, str]:
        # Only the fake API records calls. In --live mode there's nothing to
        # inspect, so we skip (count as passed).
        if not hasattr(api, "payment_calls"):
            return True, f"payment API called {n}x (skipped: live)"
        return len(api.payment_calls) == n, f"payment API called {n}x"
    return check


def last_payment_amount(amount: float) -> Check:
    # Tool-call check: passes if the most recent charge was for this amount.
    @_tool_check
    def check(message: str, agent, api) -> tuple[bool, str]:
        if not hasattr(api, "payment_calls"):
            return True, f"payment amount == {amount} (skipped: live)"
        calls = api.payment_calls
        ok = bool(calls) and calls[-1]["amount"] == amount
        return ok, f"payment amount == {amount}"
    return check


def lookup_calls(n: int) -> Check:
    # Tool-call check: passes if the lookup API was called exactly n times.
    @_tool_check
    def check(message: str, agent, api) -> tuple[bool, str]:
        if not hasattr(api, "lookup_calls"):
            return True, f"lookup API called {n}x (skipped: live)"
        return len(api.lookup_calls) == n, f"lookup API called {n}x"
    return check


def never_charged() -> Check:
    # NEGATIVE tool-call check: passes if the payment API was NEVER called.
    # Proves we don't charge before verification, or on invalid input.
    @_tool_check
    def check(message: str, agent, api) -> tuple[bool, str]:
        if not hasattr(api, "payment_calls"):
            return True, "payment API never called (skipped: live)"
        return len(api.payment_calls) == 0, "payment API never called"
    return check


def no_sensitive_leak() -> Check:
    # Security check: none of the stored secret values should appear in a reply.
    SECRETS = ["4321", "400001", "9876", "400002", "2468", "1357", "400004"]
    def check(message: str, agent, api) -> tuple[bool, str]:
        ok = all(s not in message for s in SECRETS)
        return ok, "no sensitive data leaked"
    return check


# --- the list of all test conversations ------------------------------------ #
# Each Scenario below reads like a chat script with checks attached. The comment
# above each one says what it's testing. run_eval.py plays them all.
def all_scenarios() -> list[Scenario]:
    return [
        # 1) Happy path with messy, natural-language inputs across every field.
        Scenario(
            name="happy_path_messy_inputs",
            category="happy",
            turns=[
                Turn("Hi", [msg_contains("account")]),
                Turn("yeah my account number is ACC1001 I think", [msg_contains("name")]),
                Turn("it's Nithin, Nithin Jain", [no_sensitive_leak()]),
                Turn("I was born on 14th May 1990",
                     [verified(True), msg_contains("balance", "1,250.75"), no_sensitive_leak()]),
                Turn("I want to pay a thousand rupees", [msg_contains("card")]),
                Turn(
                    "the card number is 4532 0151 1283 0366, expires December 2027, "
                    "CVV is one two three, name on card Nithin Jain",
                    [msg_contains("confirm")],
                ),
                Turn("yes", [msg_contains("recap", "transaction id"), has_transaction()]),
            ],
            final_checks=[step_is("closed_success"), verified(True),
                          payment_calls(1), last_payment_amount(1000.0)],
        ),

        # 2) Partial payment of the full balance via "clear the full amount".
        Scenario(
            name="happy_path_full_balance",
            category="happy",
            turns=[
                Turn("account id: acc1002", [msg_contains("name")]),
                Turn("Rajarajeswari Balasubramaniam", []),
                Turn("Aadhaar ends with 9876", [verified(True), no_sensitive_leak()]),
                Turn("just clear the full amount", [msg_contains("card")]),
                Turn("4532 0151 1283 0366, 12/27, cvv 123, name Raja",
                     [msg_contains("confirm")]),
                Turn("yes", [msg_contains("recap", "transaction id"), has_transaction()]),
            ],
            final_checks=[step_is("closed_success"), last_payment_amount(540.0)],
        ),

        # 3) Verification failure: wrong secondary factor, exhaust 3 retries.
        Scenario(
            name="verification_failure_exhausted",
            category="verification_failure",
            turns=[
                Turn("ACC1001", []),
                Turn("Nithin Jain", []),
                Turn("my dob is 2000-01-01", [verified(False), msg_contains("2 attempts")]),
                Turn("Nithin Jain", []),
                Turn("dob 2000-02-02", [verified(False), msg_contains("1 attempt")]),
                Turn("Nithin Jain", []),
                Turn("pincode 999999", [verified(False), msg_contains("couldn't verify")]),
            ],
            final_checks=[step_is("closed_failure"), verified(False), payment_calls(0)],
        ),

        # 4) Wrong name blocks verification even with a correct factor.
        Scenario(
            name="verification_wrong_name",
            category="verification_failure",
            turns=[
                Turn("ACC1001", []),
                Turn("Nithin Kumar", []),   # wrong surname
                Turn("dob 1990-05-14", [verified(False), no_sensitive_leak()]),
            ],
            final_checks=[verified(False), payment_calls(0)],
        ),

        # 5) Payment failure: invalid card (Luhn) caught locally, no API call.
        Scenario(
            name="payment_invalid_card_local",
            category="payment_failure",
            turns=[
                Turn("ACC1001", [lookup_calls(1)]),
                Turn("Nithin Jain", []),
                Turn("dob 1990-05-14", [verified(True)]),
                Turn("pay 500", [msg_contains("card")]),
                # Bad card (Luhn fail) -> rejected locally with a GENERIC message
                # (we don't say which field was wrong), and no API call.
                Turn("card 1234 5678 9012 3456 exp 12/2027 cvv 123 name Nithin Jain",
                     [msg_contains("couldn't be validated"), never_charged()]),
            ],
            final_checks=[verified(True), never_charged()],
        ),

        # 5b) Payment failure: expired card, caught locally (no API call).
        Scenario(
            name="payment_expired_card_local",
            category="payment_failure",
            turns=[
                Turn("ACC1001", []),
                Turn("Nithin Jain", []),
                Turn("dob 1990-05-14", [verified(True)]),
                Turn("pay 500", []),
                # Expired card -> same GENERIC "couldn't be validated" message
                # (we don't disclose that it was specifically the expiry), no API call.
                Turn("4532 0151 1283 0366, exp 01/2020, cvv 123, name Nithin Jain",
                     [msg_contains("couldn't be validated"), never_charged()]),
            ],
            final_checks=[verified(True), never_charged()],
        ),

        # 5c) Payment failure: server-side insufficient_balance, then guided
        # recovery is offered (fixable error -> re-ask amount).
        Scenario(
            name="payment_insufficient_balance_server",
            category="payment_failure",
            api_factory=_forced_error_client("insufficient_balance", 422),
            turns=[
                Turn("ACC1001", []),
                Turn("Nithin Jain", []),
                Turn("dob 1990-05-14", [verified(True)]),
                Turn("pay 500", []),
                Turn("4532 0151 1283 0366, 12/27, cvv 123, name Nithin Jain",
                     [msg_contains("confirm"), never_charged()]),  # not charged until confirmed
                Turn("yes", [msg_contains("exceeds"), payment_calls(1)]),
            ],
            # Fixable error routes back to the amount step, not a terminal close.
            final_checks=[step_is("await_amount")],
        ),

        # 5d) Payment failure: server-side invalid_amount is communicated clearly.
        Scenario(
            name="payment_invalid_amount_server",
            category="payment_failure",
            api_factory=_forced_error_client("invalid_amount", 422),
            turns=[
                Turn("ACC1001", []),
                Turn("Nithin Jain", []),
                Turn("dob 1990-05-14", [verified(True)]),
                Turn("pay 500", []),
                Turn("4532 0151 1283 0366, 12/27, cvv 123, name Nithin Jain",
                     [msg_contains("confirm"), never_charged()]),
                Turn("yes", [msg_contains("amount"), payment_calls(1)]),
            ],
            final_checks=[step_is("await_amount")],
        ),

        # 5e) Confirmation: user declines at the confirmation step -> cancelled,
        # card never charged.
        Scenario(
            name="payment_declined_at_confirmation",
            category="payment_failure",
            turns=[
                Turn("ACC1001", []),
                Turn("Nithin Jain", []),
                Turn("dob 1990-05-14", [verified(True)]),
                Turn("pay 500", []),
                Turn("4532 0151 1283 0366, 12/27, cvv 123, name Nithin Jain",
                     [msg_contains("confirm")]),
                Turn("no, cancel", [msg_contains("cancel"), never_charged()]),
            ],
            final_checks=[step_is("closed_failure"), never_charged()],
        ),

        # 6) Edge case: leap-year DOB (ACC1004, 1988-02-29) verifies exactly.
        Scenario(
            name="edge_leap_year_exact",
            category="edge_case",
            turns=[
                Turn("ACC1004", []),
                Turn("Rahul Mehta", []),
                Turn("29th February 1988", [verified(True), msg_contains("3,200.50")]),
            ],
            final_checks=[verified(True)],
        ),

        # 7) Edge case: nearby-but-wrong date (1988-02-28) must fail.
        Scenario(
            name="edge_leap_year_nearby_wrong",
            category="edge_case",
            turns=[
                Turn("ACC1004", []),
                Turn("Rahul Mehta", []),
                Turn("28-02-1988", [verified(False)]),
            ],
            final_checks=[verified(False), payment_calls(0)],
        ),

        # 7b) Edge case: zero-balance account (ACC1003) closes cleanly.
        Scenario(
            name="edge_zero_balance_nothing_to_pay",
            category="edge_case",
            turns=[
                Turn("ACC1003", []),
                Turn("Priya Agarwal", []),
                Turn("dob 1992-08-10",
                     [verified(True), msg_contains("nothing to pay")]),
            ],
            final_checks=[step_is("closed_success"), payment_calls(0)],
        ),

        # 8) Edge case: info given step-by-step (one thing per turn).
        # We intentionally capture only what the current step asks for, so
        # the account turn grabs the id, the name comes next, then a factor.
        Scenario(
            name="edge_step_by_step",
            category="edge_case",
            turns=[
                Turn("my account is ACC1001", [msg_contains("name")]),
                Turn("Nithin Jain", [msg_contains("verify")]),
                Turn("DOB is May 14, 90", [verified(True)]),
            ],
            final_checks=[verified(True)],
        ),

        # 9) Edge case: account not found, then a valid one.
        Scenario(
            name="edge_account_not_found_recovery",
            category="edge_case",
            turns=[
                Turn("ACC9999", [msg_contains("couldn't find")]),
                Turn("ACC1001", [msg_contains("name")]),
            ],
            final_checks=[],
        ),

        # 10) Edge case: user tries to pay before verification (hard rule).
        Scenario(
            name="edge_early_payment_blocked",
            category="edge_case",
            turns=[
                Turn("ACC1001", []),
                Turn("here is my card 4532 0151 1283 0366 exp 12/2027 cvv 123",
                     [verified(False), payment_calls(0)]),
            ],
            final_checks=[verified(False), payment_calls(0)],
        ),
    ]
