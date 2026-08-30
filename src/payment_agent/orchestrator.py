"""Deterministic state machine - the brain of the agent.

This module owns ALL decisions: which field is needed next, when to call each
API, whether verification passed, and how to handle every error. The LLM is
never consulted here; it only feeds structured candidates in via the extractor.

Hard rules enforced structurally:
- No payment step is reachable until `state.is_verified` is True.
- Steps are never skipped even if the user volunteers info early: out-of-order
  data is *captured* into state but the flow still advances one gate at a time.
- All inputs are validated before any API call.
- Verification is strict and deterministic (delegated to verifier.py).
- Sensitive account data is never placed into a response.
"""

from __future__ import annotations

import logging

from . import verifier
from .config import Config, DEFAULT_CONFIG
from .extractors.base import Expecting, ExtractionResult
from .responses import PAYMENT_ERROR_GUIDANCE, Responses
from .state import (
    Account,
    CardDetails,
    IdentityClaim,
    MalformedAccountError,
    SessionState,
    Step,
)
from .tools.payment_api import ApiResult, PaymentApiClient
from .validators import (
    parse_date_strict,
    validate_amount,
    validate_card_number,
    validate_cvv,
    validate_expiry,
)

logger = logging.getLogger("payment_agent.orchestrator")


# Maps the current step to what the extractor should expect.
_EXPECTING = {
    Step.GREETING: Expecting.ACCOUNT,
    Step.AWAIT_ACCOUNT: Expecting.ACCOUNT,
    Step.AWAIT_IDENTITY: Expecting.IDENTITY,
    Step.AWAIT_AMOUNT: Expecting.AMOUNT,
    Step.AWAIT_CARD: Expecting.CARD,
    Step.PROCESSING: Expecting.CARD,
}


class Orchestrator:
    def __init__(self, api: PaymentApiClient, config: Config = DEFAULT_CONFIG, metrics=None):
        self._api = api
        self._config = config
        self._metrics = metrics  # optional; None = no-op

    def _m(self, name: str) -> None:
        if self._metrics is not None:
            self._metrics.incr(name)

    def expecting_for(self, state: SessionState) -> Expecting:
        return _EXPECTING.get(state.step, Expecting.ACCOUNT)

    def handle(self, state: SessionState, extracted: ExtractionResult) -> str:
        """Advance the conversation by one turn and return the user message."""
        if state.is_terminal():
            return Responses.ALREADY_CLOSED

        if extracted.wants_to_quit:
            state.step = Step.CLOSED_FAILURE
            state.clear_card()
            return Responses.USER_QUIT

        # General ambiguity guard: if this turn contributed nothing usable to
        # the current step AND the step cannot otherwise make progress from
        # existing state, ask for clarification instead of silently looping.
        # Computed BEFORE capture mutates state.
        clarification = self._maybe_clarify(state, extracted)

        # Snapshot a progress fingerprint before capture/dispatch so we can tell
        # whether this turn actually moved the conversation forward.
        before = self._progress_fingerprint(state)

        # Capture any volunteered data into state regardless of the step. This
        # gives us out-of-order handling and "don't re-ask" for free. It does
        # NOT advance the flow past a gate on its own.
        self._capture(state, extracted)

        if clarification is not None:
            message = clarification
        elif state.step in (Step.GREETING, Step.AWAIT_ACCOUNT):
            message = self._handle_account(state)
        elif state.step == Step.AWAIT_IDENTITY:
            message = self._handle_identity(state)
        elif state.step == Step.AWAIT_AMOUNT:
            message = self._handle_amount(state)
        elif state.step in (Step.AWAIT_CARD, Step.PROCESSING):
            message = self._handle_card(state)
        else:
            message = Responses.FALLBACK

        # Liveness guard: if the turn didn't move us forward, count it; too many
        # consecutive no-progress turns closes the session so it can't loop
        # forever. A turn that already reached a terminal state is exempt.
        if not state.is_terminal():
            if self._progress_fingerprint(state) != before:
                state.no_progress_turns = 0
            else:
                state.no_progress_turns += 1
                if state.no_progress_turns >= self._config.max_no_progress_turns:
                    state.step = Step.CLOSED_FAILURE
                    state.clear_card()
                    return Responses.NO_PROGRESS

        return message

    @staticmethod
    def _progress_fingerprint(state: SessionState) -> tuple:
        """A cheap snapshot of 'how far along we are'. If it changes between the
        start and end of a turn, the turn made progress."""
        return (
            state.step,
            state.account is not None,
            state.is_verified,
            state.claim.full_name,
            state.claim.has_secondary_factor(),
            state.amount,
            state.card.card_number,
            state.card.cvv,
            state.card.expiry_month,
            state.card.cardholder_name,
        )

    # ------------------------------------------------------------------ #
    # general ambiguity guard
    # ------------------------------------------------------------------ #
    def _maybe_clarify(self, state: SessionState, ex: ExtractionResult) -> str | None:
        """Return a clarification message if this turn gave nothing usable for
        the current step and the step can't progress from existing state.

        Returns None when we DID understand something, or when the step already
        has enough to advance (so we don't nag a user who is just moving on).
        """
        step = state.step

        if step in (Step.GREETING, Step.AWAIT_ACCOUNT):
            # Usable = an account id this turn, or one already pending.
            if ex.account_id or state.pending_account_id:
                return None
            return Responses.DIDNT_UNDERSTAND_ACCOUNT

        if step == Step.AWAIT_IDENTITY:
            gave_something = bool(
                ex.full_name or ex.dob_text or ex.aadhaar_last4
                or ex.pincode or ex.unclear_secondary
            )
            # Or we already hold a claim we can act on.
            already_have = bool(
                state.claim.full_name or state.claim.has_secondary_factor()
            )
            if gave_something or already_have:
                return None
            return Responses.DIDNT_UNDERSTAND_IDENTITY

        if step == Step.AWAIT_AMOUNT:
            if ex.amount is not None or ex.pay_full_balance:
                return None
            return Responses.DIDNT_UNDERSTAND_AMOUNT

        if step in (Step.AWAIT_CARD, Step.PROCESSING):
            gave_card = any((
                ex.card_number, ex.cvv, ex.expiry_month, ex.expiry_year,
                ex.cardholder_name,
            ))
            # Or we already have some card fields collected (user is completing).
            already_have = any((
                state.card.card_number, state.card.cvv,
                state.card.expiry_month, state.card.expiry_year,
                state.card.cardholder_name,
            ))
            if gave_card or already_have:
                return None
            return Responses.DIDNT_UNDERSTAND_CARD

        return None

    # ------------------------------------------------------------------ #
    # capture volunteered data (no flow advancement)
    # ------------------------------------------------------------------ #
    def _capture(self, state: SessionState, ex: ExtractionResult) -> None:
        # Account id: stash for the lookup handler (any time it appears).
        if ex.account_id and state.account is None:
            state.pending_account_id = ex.account_id

        # Amount: stash while we're collecting it.
        if state.step == Step.AWAIT_AMOUNT:
            if ex.amount is not None:
                state.pending_amount = ex.amount
            if ex.pay_full_balance:
                state.pending_pay_full = True

        # Name may be volunteered at any time (out-of-order handling), so we
        # always capture it to avoid re-asking. It is harmless without an
        # account since verification never runs until one is loaded.
        if ex.full_name and not state.claim.full_name:
            state.claim.full_name = ex.full_name

        # Secondary factors (dob / aadhaar / pincode) are ONLY captured once an
        # account is loaded. Before that, stray digits in the account-id turn
        # (e.g. "ACC1001" -> "1001") could otherwise be mistaken for a factor
        # and pollute the claim.
        if state.account is not None:
            if ex.dob_text and not state.claim.dob:
                parsed = parse_date_strict(ex.dob_text)
                if parsed.ok:
                    state.claim.dob = parsed.value
            if ex.aadhaar_last4 and not state.claim.aadhaar_last4:
                state.claim.aadhaar_last4 = ex.aadhaar_last4
            if ex.pincode and not state.claim.pincode:
                state.claim.pincode = ex.pincode
            # Remember if the user gave an unparseable number this turn so the
            # identity handler can ask for clarification instead of ignoring it.
            state.unclear_secondary = (
                ex.unclear_secondary
                and not ex.aadhaar_last4
                and not ex.pincode
                and not ex.dob_text
            )

        # Card fields are only captured once we're actually collecting them,
        # so a card number volunteered before verification is ignored (hard
        # rule: do not act on early payment info / no skipping steps).
        if state.step in (Step.AWAIT_CARD, Step.PROCESSING):
            card = state.card
            if ex.card_number and not card.card_number:
                card.card_number = ex.card_number
            if ex.cvv and not card.cvv:
                card.cvv = ex.cvv
            if ex.expiry_month and card.expiry_month is None:
                card.expiry_month = ex.expiry_month
            if ex.expiry_year and card.expiry_year is None:
                card.expiry_year = ex.expiry_year
            if ex.cardholder_name and not card.cardholder_name:
                card.cardholder_name = ex.cardholder_name

    # ------------------------------------------------------------------ #
    # account lookup
    # ------------------------------------------------------------------ #
    def _handle_account(self, state: SessionState) -> str:
        # _capture stashed any account id it saw this turn.
        account_id = state.pending_account_id
        if not account_id:
            return Responses.ASK_ACCOUNT_AGAIN

        state.account_lookup_attempts += 1
        result = self._api.lookup_account(account_id)

        if result.ok and result.data:
            try:
                state.account = Account.from_api(result.data)
            except MalformedAccountError as exc:
                # A 200 with an unparseable body: don't crash the turn. Treat it
                # as a transient upstream problem and let the user retry.
                logger.warning("Malformed lookup response: %s", exc)
                state.pending_account_id = None
                return self._network_guard(state)
            state.step = Step.AWAIT_IDENTITY
            state.pending_account_id = None
            return self._identity_prompt(state)

        if result.network_error:
            return self._network_guard(state)

        # account_not_found or any other business error on lookup
        remaining = self._config.max_account_lookup_attempts - state.account_lookup_attempts
        if remaining <= 0:
            state.step = Step.CLOSED_FAILURE
            return Responses.account_not_found(0)
        state.pending_account_id = None
        return Responses.account_not_found(remaining)

    # ------------------------------------------------------------------ #
    # identity + verification
    # ------------------------------------------------------------------ #
    def _identity_prompt(self, state: SessionState) -> str:
        if not state.claim.full_name:
            return Responses.ASK_NAME
        if not state.claim.has_secondary_factor():
            return Responses.ASK_SECONDARY.format(name=state.claim.full_name)
        return Responses.ASK_SECONDARY_ONLY

    def _handle_identity(self, state: SessionState) -> str:
        claim = state.claim
        unclear = state.unclear_secondary
        state.unclear_secondary = False

        # Need both a name and at least one secondary factor before we judge.
        if not claim.full_name:
            return Responses.ASK_NAME_FIRST
        if not claim.has_secondary_factor():
            # If the user clearly tried to give a factor but it wasn't a valid
            # Aadhaar/pincode/DOB, ASK for a clean value (guide, don't ignore).
            if unclear:
                return Responses.UNCLEAR_SECONDARY
            return Responses.ASK_SECONDARY_ONLY

        outcome = verifier.verify(claim, state.account)
        if outcome.verified:
            self._m("verification.success")
            state.is_verified = True
            # Nothing to collect if the account is already settled. Close
            # cleanly instead of entering a payment loop where every amount is
            # rejected (any amount > 0 exceeds a 0 balance; 0 is invalid).
            if round(state.account.balance, 2) <= 0:
                state.step = Step.CLOSED_SUCCESS
                return Responses.nothing_to_pay(state.account.account_id)
            state.step = Step.AWAIT_AMOUNT
            return Responses.verified_balance(state.account.balance)

        # Failed. Count the attempt.
        self._m("verification.failure")
        state.verification_attempts += 1
        remaining = self._config.max_verification_attempts - state.verification_attempts

        if remaining <= 0:
            state.claim = IdentityClaim()
            state.step = Step.CLOSED_FAILURE
            return Responses.VERIFICATION_LOCKED

        # Graceful partial handling: if the NAME matched the account, keep it and
        # only clear the (wrong) secondary factors, then re-ask just for a
        # factor. We do not re-ask for a name the user already got right. If the
        # name itself was wrong, we can't trust the claim - clear it entirely.
        if outcome.name_matches:
            good_name = state.claim.full_name
            state.claim = IdentityClaim(full_name=good_name)
            return Responses.verification_failed_factor(remaining)

        state.claim = IdentityClaim()
        return Responses.verification_failed(remaining)

    # ------------------------------------------------------------------ #
    # amount
    # ------------------------------------------------------------------ #
    def _handle_amount(self, state: SessionState) -> str:
        pending = state.pending_amount
        pay_full = state.pending_pay_full

        if pay_full:
            pending = state.account.balance

        if pending is None:
            # No amount understood this turn. (The general clarify guard covers
            # the "nothing usable" case; this is the belt-and-braces fallback.)
            return Responses.ASK_AMOUNT

        result = validate_amount(pending, state.account.balance)
        state.pending_amount = None
        state.pending_pay_full = False
        if not result.ok:
            return result.reason

        state.amount = result.value
        state.step = Step.AWAIT_CARD
        return Responses.amount_confirmed(state.amount)

    # ------------------------------------------------------------------ #
    # card + payment
    # ------------------------------------------------------------------ #
    def _handle_card(self, state: SessionState) -> str:
        card = state.card
        if not card.is_complete():
            return Responses.ask_card_fields(card.missing_fields())

        # Validate everything locally before we ever call the API.
        num = validate_card_number(card.card_number)
        if not num.ok:
            card.card_number = None
            return num.reason
        card.card_number = num.value  # normalised digits

        cvv = validate_cvv(card.cvv, card.card_number)
        if not cvv.ok:
            card.cvv = None
            return cvv.reason

        exp = validate_expiry(card.expiry_month, card.expiry_year)
        if not exp.ok:
            card.expiry_month = None
            card.expiry_year = None
            return exp.reason

        # All valid -> process payment.
        state.step = Step.PROCESSING
        return self._process_payment(state)

    def _process_payment(self, state: SessionState) -> str:
        card = state.card
        result = self._api.process_payment(
            account_id=state.account.account_id,
            amount=state.amount,
            cardholder_name=card.cardholder_name,
            card_number=card.card_number,
            cvv=card.cvv,
            expiry_month=card.expiry_month,
            expiry_year=card.expiry_year,
        )

        # PCI hygiene: never retain the CVV after an authorisation attempt. We
        # drop it immediately regardless of outcome. On a retry (e.g. the amount
        # was wrong) the card number/expiry/name are kept for convenience but the
        # CVV is re-requested, so it is never held longer than a single attempt.
        card.cvv = None

        if result.ok and result.data:
            transaction_id = result.data.get("transaction_id")
            # Defensive: a 200 that claims success but carries no transaction id
            # is a malformed/ambiguous response. We must NOT show the user a
            # "Transaction ID: None" recap or claim success we can't evidence.
            # Treat it as a transient failure and let them retry.
            if not transaction_id:
                logger.warning("Payment 200 without transaction_id: %s", result.data)
                state.step = Step.AWAIT_CARD
                return self._network_guard(state, during_payment=True)

            state.transaction_id = transaction_id
            amount = state.amount
            account_id = state.account.account_id
            # The API does not persist balance, so we recap the remaining
            # balance for THIS session as (original balance - amount paid).
            remaining_balance = round(state.account.balance - amount, 2)
            self._m("payment.success")
            state.step = Step.CLOSED_SUCCESS
            state.clear_card()
            return Responses.payment_success(
                amount, state.transaction_id, account_id, remaining_balance
            )

        if result.network_error:
            state.step = Step.AWAIT_CARD  # let them retry once system is back
            return self._network_guard(state, during_payment=True)

        return self._handle_payment_error(state, result)

    def _handle_payment_error(self, state: SessionState, result: ApiResult) -> str:
        self._m(f"payment.failure.{result.error_code or 'unknown'}")
        state.payment_attempts += 1
        remaining = self._config.max_payment_attempts - state.payment_attempts
        code = result.error_code or "invalid_args"
        reason, terminal = PAYMENT_ERROR_GUIDANCE.get(
            code, ("The payment couldn't be processed.", False)
        )

        if terminal:
            state.step = Step.CLOSED_FAILURE
            state.clear_card()
            return Responses.payment_terminal(reason)

        if remaining <= 0:
            state.step = Step.CLOSED_FAILURE
            state.clear_card()
            return Responses.payment_terminal(reason)

        # Fixable: route the user back to fix the relevant thing.
        if code == "insufficient_balance" or code == "invalid_amount":
            state.step = Step.AWAIT_AMOUNT
            state.amount = None
        else:  # card-related
            state.step = Step.AWAIT_CARD
            state.card = CardDetails()
        return Responses.payment_retryable(reason, remaining)

    # ------------------------------------------------------------------ #
    # shared network handling
    # ------------------------------------------------------------------ #
    def _network_guard(self, state: SessionState, during_payment: bool = False) -> str:
        # Count network trouble against the relevant attempt budget so we don't
        # loop forever if the API is down.
        state.network_failures += 1
        if state.network_failures >= self._config.network_max_retries + 2:
            state.step = Step.CLOSED_FAILURE
            state.clear_card()
            return Responses.NETWORK_ERROR_TERMINAL
        return Responses.NETWORK_ERROR
