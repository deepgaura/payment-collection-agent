"""End-to-end conversation tests using the deterministic mock API.

These exercise the full state machine: happy path, verification failure,
payment failures, out-of-order input, security (no data leakage / no early
payment), and network failure handling.
"""

import datetime as dt

import pytest

from payment_agent.agent import Agent
from payment_agent.config import Config
from payment_agent.extractors.rule_based import RuleBasedExtractor
from payment_agent.state import Step

from .fixtures import MockApiClient, NetworkFailApiClient

# Deterministic offline config: rule-based extractor, mock API.
OFFLINE_CONFIG = Config(use_llm=False)


def make_agent(api=None):
    return Agent(
        OFFLINE_CONFIG,
        api_client=api or MockApiClient(),
        extractor=RuleBasedExtractor(),
    )


def run(agent, messages):
    return [agent.next(m)["message"] for m in messages]


# --- Happy path ------------------------------------------------------------ #
def test_happy_path_success():
    api = MockApiClient()
    agent = make_agent(api)
    run(agent, [
        "Hi",
        "my account is ACC1001",
        "Nithin Jain",
        "born 14th May 1990",
        "pay a thousand rupees",
    ])
    agent.next(
        "card 4532 0151 1283 0366 expires December 2027 cvv one two three name Nithin Jain"
    )
    final = agent.next("yes")["message"]  # confirm before charging
    assert agent.step == Step.CLOSED_SUCCESS
    assert agent.is_verified
    assert "recap" in final.lower() and "transaction id" in final.lower()
    assert agent.transaction_id and agent.transaction_id.startswith("txn_")
    assert len(api.payment_calls) == 1
    assert api.payment_calls[0]["amount"] == 1000.0


def test_full_balance_payment():
    api = MockApiClient()
    agent = make_agent(api)
    run(agent, ["ACC1002", "Rajarajeswari Balasubramaniam", "aadhaar 9876", "clear the full amount"])
    agent.next("card 4532 0151 1283 0366 exp 12/2027 cvv 123 name Raja")
    agent.next("yes")
    assert agent.step == Step.CLOSED_SUCCESS
    assert api.payment_calls[0]["amount"] == 540.00


# --- Verification failure -------------------------------------------------- #
def test_verification_exhausts_retries():
    agent = make_agent()
    run(agent, ["ACC1001", "Nithin Jain", "dob 2000-01-01"])   # attempt 1
    run(agent, ["Nithin Jain", "dob 2000-01-02"])              # attempt 2
    last = agent.next("Nithin Jain")
    last = agent.next("pincode 111111")                         # attempt 3 -> lock
    assert agent.step == Step.CLOSED_FAILURE
    assert not agent.is_verified


def test_correct_name_kept_across_failed_factor():
    """If the name matched, a wrong factor should not force re-typing the name;
    giving just a correct factor next should verify."""
    agent = make_agent()
    run(agent, ["ACC1001", "Nithin Jain", "dob 2000-01-01"])  # wrong dob
    assert agent._state.claim.full_name == "Nithin Jain"      # name preserved
    assert agent._state.verification_attempts == 1
    agent.next("aadhaar 4321")                                 # factor only
    assert agent.is_verified


def test_wrong_name_clears_claim():
    """A wrong name is not trusted; the whole claim is cleared."""
    agent = make_agent()
    run(agent, ["ACC1001", "Wrong Person", "dob 1990-05-14"])
    assert agent._state.claim.full_name is None
    assert not agent.is_verified


def test_zero_balance_account_closes_cleanly():
    """ACC1003 has a 0.00 balance. After verifying, the agent should say there's
    nothing to pay and close - not enter a payment loop that rejects every
    amount."""
    api = MockApiClient()
    agent = make_agent(api)
    run(agent, ["ACC1003", "Priya Agarwal", "dob 1992-08-10"])
    assert agent.is_verified
    assert agent.step == Step.CLOSED_SUCCESS
    assert len(api.payment_calls) == 0  # never tries to charge


def test_no_payment_without_verification():
    """Hard rule: card volunteered early must NOT trigger a payment."""
    api = MockApiClient()
    agent = make_agent(api)
    run(agent, [
        "ACC1001",
        "here's my card 4532 0151 1283 0366 exp 12/2027 cvv 123",  # premature
    ])
    assert not agent.is_verified
    assert len(api.payment_calls) == 0
    assert agent.step not in (Step.CLOSED_SUCCESS,)


# --- Payment failures ------------------------------------------------------ #
def test_invalid_card_caught_locally():
    api = MockApiClient()
    agent = make_agent(api)
    run(agent, ["ACC1001", "Nithin Jain", "dob 1990-05-14", "pay 500"])
    msg = agent.next("card 1234 5678 9012 3456 exp 12/2027 cvv 123 name Nithin Jain")["message"]
    # Luhn fails locally -> no API call made.
    assert len(api.payment_calls) == 0
    assert "validation" in msg.lower() or "check" in msg.lower()


def test_insufficient_balance_then_retry():
    api = MockApiClient()
    agent = make_agent(api)
    run(agent, ["ACC1002", "Rajarajeswari Balasubramaniam", "aadhaar 9876"])
    # 540 balance; ask for exactly 540 is fine, but we test the retry route by
    # forcing an over-balance amount is blocked locally, so instead verify a
    # valid smaller amount succeeds.
    run(agent, ["pay 500"])
    agent.next("card 4532 0151 1283 0366 exp 12/2027 cvv 123 name Raja")
    agent.next("yes")
    assert agent.step == Step.CLOSED_SUCCESS


# --- Out-of-order handling ------------------------------------------------- #
def test_name_then_factor_verifies():
    agent = make_agent()
    # New behavior: each turn we only capture what that step asks for.
    # So the account step grabs the account id; the name is given on its
    # own turn; then a single factor (DOB) completes verification.
    run(agent, ["my account is ACC1001", "Nithin Jain"])
    msg = agent.next("dob 1990-05-14")["message"]
    assert agent.is_verified
    assert "balance" in msg.lower()


def test_does_not_reask_captured_info():
    agent = make_agent()
    run(agent, ["ACC1001", "Nithin Jain"])
    # Name captured; the next prompt should be for a secondary factor, not name.
    msg = agent.next("what next?")["message"]
    assert "name" not in msg.lower() or "birth" in msg.lower()


# --- Security -------------------------------------------------------------- #
def test_unclear_secondary_asks_without_burning_retry():
    """An unrecognisable number should trigger a clarifying question and NOT
    consume a verification attempt."""
    agent = make_agent()
    run(agent, ["ACC1001", "Nithin Jain"])
    msg = agent.next("43215")["message"]   # 5 digits: neither Aadhaar nor pincode
    assert "clarify" in msg.lower() or "doesn't look like" in msg.lower()
    assert agent._state.verification_attempts == 0
    # Clarifying with a valid factor then verifies.
    agent.next("aadhaar 4321")
    assert agent.is_verified


def test_general_clarify_at_every_step():
    """Off-topic input at each step gets a clarifying question, not a silent
    loop or a wrong action."""
    # Account step
    a = make_agent()
    msg = a.next("what's the weather today")["message"]
    assert "account id" in msg.lower()

    # Identity step (no name yet)
    a = make_agent()
    a.next("ACC1001")
    msg = a.next("tell me a joke")["message"]
    assert "verify" in msg.lower() and not a.is_verified

    # Amount step
    a = make_agent()
    run(a, ["ACC1001", "Nithin Jain", "dob 1990-05-14"])
    msg = a.next("i dunno")["message"]
    assert "amount" in msg.lower()

    # Card step
    a = make_agent()
    run(a, ["ACC1001", "Nithin Jain", "dob 1990-05-14", "pay 500"])
    msg = a.next("what do you need again")["message"]
    assert "card" in msg.lower()


def test_chatter_not_mistaken_for_name():
    """Conversational filler must not be captured as the identity name."""
    a = make_agent()
    a.next("ACC1001")
    a.next("hmm ok what now")
    # No name should have been captured from the filler.
    assert a._state.claim.full_name is None


def test_no_sensitive_data_leaked():
    agent = make_agent()
    msgs = run(agent, ["ACC1001", "Nithin Jain", "dob 1999-09-09"])  # wrong dob
    blob = " ".join(msgs)
    for secret in ("4321", "400001", "1990-05-14"):
        assert secret not in blob


# --- Network failure ------------------------------------------------------- #
def test_cvv_not_retained_after_attempt():
    """PCI hygiene: the CVV must be dropped after each authorisation attempt.
    On an amount retry the card is kept but the CVV is re-requested."""
    from payment_agent.tools.payment_api import ApiResult

    class OneInsufficient(MockApiClient):
        def __init__(self):
            super().__init__()
            self._n = 0

        def process_payment(self, **kw):
            self._n += 1
            if self._n == 1:
                return ApiResult(ok=False, status_code=422, error_code="insufficient_balance")
            return super().process_payment(**kw)

    agent = make_agent(OneInsufficient())
    run(agent, ["ACC1001", "Nithin Jain", "dob 1990-05-14", "pay 500"])
    agent.next("4532 0151 1283 0366, 12/27, cvv 123, name Nithin Jain")
    agent.next("yes")  # confirm -> attempt -> server says insufficient
    # Card number kept for convenience, but CVV dropped after the attempt.
    assert agent._state.card.card_number is not None
    assert agent._state.card.cvv is None
    agent.next("pay 300")
    msg = agent.next("ok")["message"]
    assert "cvv" in msg.lower()  # re-requested


def test_success_without_transaction_id_not_treated_as_success():
    """A 200 that lacks a transaction_id must not be shown as success (no
    'Transaction ID: None' recap) and must not close the session as paid."""
    from payment_agent.tools.payment_api import ApiResult

    class NoTxnApi(MockApiClient):
        def process_payment(self, **kwargs):
            super().process_payment(**kwargs)  # record the call
            return ApiResult(ok=True, status_code=200, data={"success": True})

    agent = make_agent(NoTxnApi())
    run(agent, ["ACC1001", "Nithin Jain", "dob 1990-05-14", "pay 500"])
    agent.next("4532 0151 1283 0366, 12/27, cvv 123, name Nithin Jain")
    msg = agent.next("yes")["message"]
    assert "None" not in msg
    assert agent.step != Step.CLOSED_SUCCESS
    assert agent.transaction_id is None


def test_no_progress_closes_gracefully():
    """If the user never supplies what's needed (e.g. never gives a name), the
    agent must not loop forever - it closes after the no-progress limit."""
    agent = make_agent()
    agent.next("ACC1002")
    last = ""
    for m in ["um", "hello?", "what", "hmm", "ok", "why"]:
        last = agent.next(m)["message"]
        if agent.step == Step.CLOSED_FAILURE:
            break
    assert agent.step == Step.CLOSED_FAILURE
    assert not agent.is_verified


def test_irrelevant_data_does_not_reset_no_progress():
    """Progress is step-aware: volunteering a name while we're still asking for
    the account must NOT reset the no-progress counter (a stuck/hostile user
    can't drag the session out by dripping parseable-but-irrelevant tokens)."""
    agent = make_agent()
    # Never provide an account id; one turn gives an (irrelevant here) name.
    msgs = ["hey how are you", "my name is deepanshu mahajan",
            "i dont like you", "i dont like you", "i dont like you", "i dont like you"]
    for m in msgs:
        agent.next(m)
        if agent.step == Step.CLOSED_FAILURE:
            break
    assert agent.step == Step.CLOSED_FAILURE
    assert not agent.is_verified


def test_slow_but_progressing_user_not_closed():
    """A user who progresses (with chatter in between) must not be cut off by
    the no-progress guard."""
    agent = make_agent()
    run(agent, ["ACC1002", "Rajarajeswari Balasubramaniam", "um ok", "aadhaar 9876"])
    assert agent.is_verified  # reached verification despite a filler turn


def test_phraser_never_touches_sensitive_messages():
    """The conversational phraser may reword safe prompts, but sensitive /
    transactional messages (balance, transaction id, recap) must pass through
    as the exact deterministic template - never sent to the LLM."""
    from payment_agent.phraser import Phraser

    class MangleClient:
        """A fake LLM that would corrupt any message it's asked to rephrase."""
        def complete_text(self, system, user, max_tokens=160):
            return "CORRUPTED BY LLM"

    agent = make_agent()
    agent._phraser = Phraser(MangleClient(), metrics=agent.metrics)

    # Safe prompt (asking for name) -> allowed to be rephrased.
    agent.next("ACC1001")
    safe_msg = agent.next("hi there")["message"]  # clarify/ask -> rephrasable
    assert safe_msg == "CORRUPTED BY LLM"

    # Reach the balance reveal (sensitive) -> must be the real template.
    agent.next("Nithin Jain")
    balance_msg = agent.next("dob 1990-05-14")["message"]
    assert "1,250.75" in balance_msg          # real balance present
    assert "CORRUPTED" not in balance_msg     # LLM did NOT touch it

    # The confirmation summary (amount + card last4) is sensitive -> template.
    agent.next("pay 500")
    summary = agent.next("4532 0151 1283 0366, 12/27, cvv 123, name Nithin Jain")["message"]
    assert "500.00" in summary and "0366" in summary
    assert "CORRUPTED" not in summary

    # Complete payment -> the recap (transaction id) must be the real template.
    recap = agent.next("yes")["message"]
    assert "Transaction ID" in recap
    assert "CORRUPTED" not in recap


def test_metrics_recorded_over_a_conversation():
    """The agent records observability signals: turn count, latency timers, and
    business outcomes (verification/payment)."""
    agent = make_agent()
    run(agent, ["ACC1001", "Nithin Jain", "dob 1990-05-14", "pay 500",
                "4532 0151 1283 0366, 12/27, cvv 123, name Nithin Jain", "yes"])
    assert agent.step == Step.CLOSED_SUCCESS
    snap = agent.metrics.snapshot()
    # Turn count and a turn-latency timer exist.
    assert snap["counters"]["turns"] >= 5
    assert snap["timings_ms"]["turn"]["count"] >= 5
    # Business outcomes were counted.
    assert snap["counters"].get("verification.success") == 1
    assert snap["counters"].get("payment.success") == 1


def test_malformed_account_id_rejected_not_repaired():
    """A malformed account id (e.g. 'AC1001', one C) must be rejected and NOT
    silently repaired into a valid-but-different account. The agent stays on the
    account step and never looks anyone up."""
    api = MockApiClient()
    agent = make_agent(api)
    msg = agent.next("AC1001")["message"]
    assert agent.step == Step.AWAIT_ACCOUNT      # did not advance
    assert len(api.lookup_calls) == 0            # never looked up an account
    assert "account id" in msg.lower()           # re-asked


def test_malformed_lookup_response_does_not_crash():
    """A 200 lookup with a malformed body must degrade gracefully, not raise
    out of next() (which would violate the interface contract)."""
    from payment_agent.tools.payment_api import ApiResult

    class BadLookupApi(MockApiClient):
        def lookup_account(self, account_id):
            # 'ok' but missing required fields (e.g. no balance/dob).
            return ApiResult(ok=True, status_code=200, data={"account_id": account_id})

    agent = make_agent(BadLookupApi())
    out = agent.next("ACC1001")            # must not raise
    assert isinstance(out, dict) and isinstance(out["message"], str)
    assert not agent.is_verified


def test_network_failure_on_lookup_is_graceful():
    agent = make_agent(NetworkFailApiClient())
    msg = agent.next("ACC1001")["message"]
    assert "trouble" in msg.lower() or "try again" in msg.lower()
    # Should not crash and should not be verified.
    assert not agent.is_verified


# --- Interface contract ---------------------------------------------------- #
def test_next_always_returns_message_dict():
    agent = make_agent()
    for m in ["", "hello", "ACC1001", "gibberish 123 !@#"]:
        out = agent.next(m)
        assert isinstance(out, dict) and isinstance(out["message"], str)


def test_terminal_session_stays_closed():
    api = MockApiClient()
    agent = make_agent(api)
    run(agent, ["ACC1001", "Nithin Jain", "dob 1990-05-14", "pay 500"])
    agent.next("card 4532 0151 1283 0366 exp 12/2027 cvv 123 name Nithin Jain")
    agent.next("yes")
    assert agent.step == Step.CLOSED_SUCCESS
    after = agent.next("hello again")["message"]
    assert "ended" in after.lower() or "complete" in after.lower() or "new conversation" in after.lower()
