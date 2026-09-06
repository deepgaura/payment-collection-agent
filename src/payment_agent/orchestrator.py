"""
WHAT THIS FILE IS (in one line):
    The BRAIN. It runs the conversation, decides what to do each turn, and calls
    all the other helpers. Every real decision lives here.

HOW ONE TURN FLOWS THROUGH THE APP (the big picture):
    user types something
        -> agent.py cleans it and asks the EXTRACTOR to pull out fields
           (extractors/llm.py or rule_based.py)
        -> agent.py hands those fields to THIS file's handle() method
        -> handle() figures out the current step and:
              * uses VALIDATORS (validators.py) to check card/amount/date
              * uses the VERIFIER (verifier.py) to check identity
              * calls the API CLIENT (tools/payment_api.py) to look up / charge
              * picks the reply text from RESPONSES (responses.py)
              * reads/writes the notebook (state.py -> SessionState)
        -> returns (message, intent); agent.py may soften the tone via phraser.py

THE HARD SAFETY RULES (all enforced by the code here, not by hoping):
    - You CANNOT reach a payment step until state.is_verified is True.
    - Info given early is remembered, but steps are never SKIPPED.
    - Every input is checked BEFORE any API call.
    - Identity checking is strict (done in verifier.py).
    - The stored account details are never put into a reply.

HOW TO READ THIS FILE:
    Start at handle(). It's the "traffic controller". Based on the current step
    it calls one small handler: _handle_account, _handle_identity, _handle_amount,
    _handle_card, _handle_confirmation. Read those in order and you've read the
    whole flow.
"""

from __future__ import annotations

import logging

# The other pieces the brain leans on:
from . import verifier                                # checks identity (strict)
from .config import Config, DEFAULT_CONFIG            # settings (retry limits etc.)
from .extractors.base import Expecting, ExtractionResult
from .responses import PAYMENT_ERROR_GUIDANCE, Responses   # the words we say
from .state import (                                  # the "notebook" + data types
    Account,
    CardDetails,
    IdentityClaim,
    MalformedAccountError,
    SessionState,
    Step,
)
from .tools.payment_api import ApiResult, PaymentApiClient  # talks to the real API
from .validators import (                             # input checkers
    normalize_account_id,
    parse_date_strict,
    validate_amount,
    validate_card_number,
    validate_cvv,
    validate_expiry,
)

logger = logging.getLogger("payment_agent.orchestrator")


# A lookup table: "given the current step, what should the extractor look for?"
# Example: while at AWAIT_AMOUNT we tell the extractor to expect an AMOUNT, so
# it reads "500" as money. (This is used by expecting_for below.)
_EXPECTING = {
    Step.GREETING: Expecting.ACCOUNT,
    Step.AWAIT_ACCOUNT: Expecting.ACCOUNT,
    Step.AWAIT_IDENTITY: Expecting.IDENTITY,
    Step.AWAIT_AMOUNT: Expecting.AMOUNT,
    Step.AWAIT_CARD: Expecting.CARD,
    Step.AWAIT_CONFIRMATION: Expecting.CONFIRMATION,
    Step.PROCESSING: Expecting.CARD,
}


class Orchestrator:
    def __init__(self, api: PaymentApiClient, config: Config = DEFAULT_CONFIG, metrics=None):
        self._api = api            # the API client (lookup + charge)
        self._config = config      # settings (retry limits, etc.)
        self._metrics = metrics    # optional counter for stats; None = ignore

    def _m(self, name: str) -> None:
        # Tiny helper: bump a metric counter if metrics are turned on.
        if self._metrics is not None:
            self._metrics.incr(name)

    def expecting_for(self, state: SessionState) -> Expecting:
        # "Given where we are, what should the extractor look for?" (uses the
        # _EXPECTING table above). agent.py calls this before extracting.
        return _EXPECTING.get(state.step, Expecting.ACCOUNT)

    def handle(self, state: SessionState, extracted: ExtractionResult) -> tuple[str, str]:
        """
        THE MAIN METHOD - runs ONE turn of the conversation.

        Inputs:
          state     = the notebook (what we know so far)
          extracted = the clean fields the extractor pulled from this message

        Returns: (message, intent)
          message = the text to show the user
          intent  = a label saying how "sensitive" the message is. The phraser
                    uses it to decide if it's allowed to reword the message.
                    Sensitive ones (balance, recap, errors) get "unsafe"/"closed"
                    so they're shown EXACTLY as written.
        """
        # If the session already ended, just say so and stop.
        if state.is_terminal():
            return Responses.ALREADY_CLOSED, "closed"

        # If the user said "cancel/quit", end cleanly and wipe card data.
        if extracted.wants_to_quit:
            state.step = Step.CLOSED_FAILURE
            state.clear_card()
            return Responses.USER_QUIT, "closed"

        # Remember if we were ALREADY verified when this turn began. This helps
        # us tell "just asking for the amount" (safe to reword) apart from the
        # "verified! your balance is X" message (sensitive - never reword).
        was_verified = state.is_verified

        # Did the user give us nothing usable? If so, prepare a "didn't catch
        # that" message (computed now, used below).
        clarification = self._maybe_clarify(state, extracted)

        # Take a snapshot of "how far along are we" BEFORE we save the new input.
        # We compare before/after to detect a stuck user (see liveness guard).
        start_step = state.step
        before = self._step_progress(state, start_step)

        # Save whatever useful data the user gave (name, dob, card fields...).
        # NOTE: saving data does NOT skip steps - it just avoids re-asking.
        self._capture(state, extracted)

        # --- Now decide what to say, based on the current step. ---
        if clarification is not None:
            # We didn't understand this turn -> ask for clarification.
            message = clarification
            intent = "clarify"
        elif state.step in (Step.GREETING, Step.AWAIT_ACCOUNT):
            # We're collecting the account id -> look it up.
            message = self._handle_account(state)
            # After this, we either moved to identity (safe "what's your name")
            # or we're still asking for the account (also safe to reword).
            intent = "ask_identity" if state.step == Step.AWAIT_IDENTITY else "ask_account"
        elif state.step == Step.AWAIT_IDENTITY:
            # We're verifying identity.
            message = self._handle_identity(state)
            # If verification just PASSED, the reply reveals the balance
            # (sensitive) -> "unsafe" so the phraser won't touch it. Otherwise
            # it's just asking for identity (safe... but we keep identity prompts
            # as plain templates anyway, see phraser.SAFE_INTENTS).
            intent = "unsafe" if (state.is_verified and not was_verified) else "ask_identity"
        elif state.step == Step.AWAIT_AMOUNT:
            # We're collecting how much to pay.
            message = self._handle_amount(state)
            # The FIRST time we land here the message includes the balance line
            # (sensitive). After that it's just "how much?" (safe).
            intent = "unsafe" if (state.is_verified and not was_verified) else "ask_amount"
        elif state.step == Step.AWAIT_CONFIRMATION:
            # We're waiting for the user's yes/no before charging.
            message = self._handle_confirmation(state, extracted)
            # If still awaiting confirmation, it's just a re-ask (safe). If we
            # moved on (charged/cancelled), that's sensitive -> "unsafe".
            intent = "clarify" if state.step == Step.AWAIT_CONFIRMATION else "unsafe"
        elif state.step in (Step.AWAIT_CARD, Step.PROCESSING):
            # We're collecting card fields.
            message = self._handle_card(state)
            # Asking for a card field is safe. But once the card is complete we
            # move to the confirmation SUMMARY (amount + card last4), which is
            # sensitive -> "unsafe".
            intent = "ask_card" if state.step == Step.AWAIT_CARD else "unsafe"
        else:
            # Shouldn't normally happen; a safe generic reply.
            message = Responses.FALLBACK
            intent = "clarify"

        # --- LIVENESS GUARD: don't let a stuck user loop forever. ---
        # We compare "how far along" now vs the snapshot from the turn's start.
        # If nothing moved (same step, no new relevant data), count a
        # "no-progress" turn. Too many in a row -> close the session cleanly.
        if not state.is_terminal():
            advanced = state.step != start_step            # did we change step?
            if advanced or self._step_progress(state, start_step) != before:
                state.no_progress_turns = 0                # made progress -> reset
            else:
                state.no_progress_turns += 1               # stuck -> count it
                if state.no_progress_turns >= self._config.max_no_progress_turns:
                    state.step = Step.CLOSED_FAILURE
                    state.clear_card()
                    return Responses.NO_PROGRESS, "closed"

        # If this turn ended the session (success or failure), its message is
        # transactional/sensitive -> never let the phraser reword it.
        if state.is_terminal():
            intent = "closed"

        return message, intent

    @staticmethod
    def _step_progress(state: SessionState, step: Step) -> tuple:
        """
        Return a small "how far along are we?" snapshot for the CURRENT step
        ONLY. The liveness guard compares this before vs after a turn.

        Why "current step only"? So that giving the wrong thing doesn't count as
        progress. Example: while we're asking for the ACCOUNT, if the user only
        says their name, that name is NOT what this step needs -> no progress ->
        the stuck-counter keeps rising (so a hostile user can't stall forever).
        """
        if step in (Step.GREETING, Step.AWAIT_ACCOUNT):
            return (state.account is not None,)
        if step == Step.AWAIT_IDENTITY:
            return (
                state.is_verified,
                state.claim.full_name,
                state.claim.has_secondary_factor(),
            )
        if step == Step.AWAIT_AMOUNT:
            return (state.amount,)
        if step in (Step.AWAIT_CARD, Step.PROCESSING):
            return (
                state.card.card_number,
                state.card.cvv,
                state.card.expiry_month,
                state.card.expiry_year,
                state.card.cardholder_name,
            )
        if step == Step.AWAIT_CONFIRMATION:
            # Progress here is leaving the step (handled by the step-change
            # check); repeated ambiguous replies count as no-progress.
            return (state.step,)
        return ()

    # ------------------------------------------------------------------ #
    # general ambiguity guard
    # ------------------------------------------------------------------ #
    def _maybe_clarify(self, state: SessionState, ex: ExtractionResult) -> str | None:
        """
        "Did we understand anything useful this turn?"

        Returns a friendly "I didn't catch that, here's what I need" message if
        the user gave nothing usable for the current step. Returns None if we DID
        get something (or already have enough), so we don't nag the user.

        This is what turns rude silence into a helpful re-prompt at every step.
        """
        step = state.step

        # ACCOUNT step: usable = an account id (now or already pending).
        if step in (Step.GREETING, Step.AWAIT_ACCOUNT):
            if ex.account_id or state.pending_account_id:
                return None
            return Responses.DIDNT_UNDERSTAND_ACCOUNT

        # IDENTITY step: usable = a name/dob/aadhaar/pincode this turn, OR we
        # already have some of the claim saved.
        if step == Step.AWAIT_IDENTITY:
            gave_something = bool(
                ex.full_name or ex.dob_text or ex.aadhaar_last4
                or ex.pincode or ex.unclear_secondary
            )
            already_have = bool(
                state.claim.full_name or state.claim.has_secondary_factor()
            )
            if gave_something or already_have:
                return None
            return Responses.DIDNT_UNDERSTAND_IDENTITY

        # AMOUNT step: usable = an amount or "pay the full balance".
        if step == Step.AWAIT_AMOUNT:
            if ex.amount is not None or ex.pay_full_balance:
                return None
            return Responses.DIDNT_UNDERSTAND_AMOUNT

        # CARD step: usable = any card field this turn, OR we already have some.
        if step in (Step.AWAIT_CARD, Step.PROCESSING):
            gave_card = any((
                ex.card_number, ex.cvv, ex.expiry_month, ex.expiry_year,
                ex.cardholder_name,
            ))
            already_have = any((
                state.card.card_number, state.card.cvv,
                state.card.expiry_month, state.card.expiry_year,
                state.card.cardholder_name,
            ))
            if gave_card or already_have:
                return None
            return Responses.DIDNT_UNDERSTAND_CARD

        # Other steps (e.g. confirmation) handle their own re-asking.
        return None

    # ------------------------------------------------------------------ #
    # capture volunteered data (no flow advancement)
    # ------------------------------------------------------------------ #
    def _capture(self, state: SessionState, ex: ExtractionResult) -> None:
        """
        Save any useful data the user gave into the notebook (state).

        KEY IDEA: saving is generous, but it does NOT advance the flow. We store
        what the user said so we never re-ask - but the step only moves forward
        when its own handler decides it's ready. That's how "don't re-ask" and
        "don't skip steps" both hold at once.
        """
        # Account id: remember it for the lookup step (can appear any time).
        # Deterministic shape guard: only accept a well-formed "ACC<digits>" id
        # and take it EXACTLY as given (case/space normalised only). This stops
        # a misbehaving LLM from "repairing" a typo like "AC1001" into a valid
        # but different account. A malformed id is dropped -> the agent re-asks.
        if ex.account_id and state.account is None:
            canonical = normalize_account_id(ex.account_id)
            if canonical:
                state.pending_account_id = canonical

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
        """
        STEP: look up the account.
        Reads the pending account id, calls the API, and:
          - found      -> move to identity, ask for the name
          - not found  -> re-ask (up to a limit, then close)
          - API down   -> graceful "try again" message
        """
        # _capture already saved a valid account id here (or None if not given).
        account_id = state.pending_account_id
        if not account_id:
            return Responses.ASK_ACCOUNT_AGAIN

        state.account_lookup_attempts += 1
        result = self._api.lookup_account(account_id)   # <-- real API call

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
        # Pick the right "please give me..." message based on what's still missing.
        if not state.claim.full_name:
            return Responses.ASK_NAME                                   # need name
        if not state.claim.has_secondary_factor():
            return Responses.ASK_SECONDARY.format(name=state.claim.full_name)  # need a factor
        return Responses.ASK_SECONDARY_ONLY

    def _handle_identity(self, state: SessionState) -> str:
        """
        STEP: verify the user.
        We need BOTH a name and one factor (dob/aadhaar/pincode). Once we have
        both, we run the strict check (verifier.verify). Pass -> show balance.
        Fail -> count an attempt, keep a correct name, and re-ask (up to 3).
        """
        claim = state.claim
        unclear = state.unclear_secondary   # did they give a junk number this turn?
        state.unclear_secondary = False

        # Can't judge yet if we're missing a piece - ask for it.
        if not claim.full_name:
            return Responses.ASK_NAME_FIRST
        if not claim.has_secondary_factor():
            # They typed a number that wasn't a valid aadhaar/pincode/dob ->
            # ask them to clarify (rather than silently ignoring it).
            if unclear:
                return Responses.UNCLEAR_SECONDARY
            return Responses.ASK_SECONDARY_ONLY

        # We have name + factor -> do the strict check (in verifier.py).
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
        """
        STEP: collect the payment amount.
        Reads the amount the user gave, checks it (validators.validate_amount),
        and on success moves to collecting the card.
        """
        pending = state.pending_amount
        pay_full = state.pending_pay_full

        # "pay the full amount" -> use the exact outstanding balance.
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
        """
        STEP: collect and check the card.
        Asks for card fields ONE AT A TIME. Once all four are in, it validates
        them locally (number/CVV/expiry). If all good, it does NOT charge yet -
        it moves to the confirmation step and shows a safe summary.
        """
        card = state.card
        if not card.is_complete():
            # Still missing something -> ask for just the next field.
            return Responses.ask_card_field(card.next_missing_field())

        # All four fields are in. Validate them together (number/expiry/CVV).
        # SECURITY CHOICE: if ANY check fails we do NOT reveal which one - we
        # clear the whole card and show a single generic "couldn't validate,
        # re-enter" message. Naming the bad field would help a card-tester probe
        # which parts of a stolen card are valid. These are format checks only,
        # so nothing stored/secret is exposed either way.
        num = validate_card_number(card.card_number)
        cvv = validate_cvv(card.cvv, card.card_number) if num.ok else None
        exp = validate_expiry(card.expiry_month, card.expiry_year)
        if not num.ok or cvv is None or not cvv.ok or not exp.ok:
            state.clear_card()   # wipe everything; user re-enters the full set
            return Responses.CARD_DETAILS_INVALID
        card.card_number = num.value  # normalised digits

        # All valid -> ask for explicit confirmation before charging (no charge
        # happens yet). The summary shows only a safe subset (amount + card
        # last 4 + expiry), never the full number or CVV.
        state.step = Step.AWAIT_CONFIRMATION
        return Responses.confirm_payment(
            state.amount, card.card_number[-4:], card.expiry_month, card.expiry_year
        )

    def _handle_confirmation(self, state: SessionState, ex: ExtractionResult) -> str:
        """
        STEP: the yes/no before charging.
          yes       -> actually charge the card
          no        -> cancel cleanly, wipe the card, don't charge
          anything else (unclear) -> re-ask; NEVER charge on a guess (safe default)
        """
        if ex.confirm is True:
            state.step = Step.PROCESSING
            return self._process_payment(state)
        if ex.confirm is False:
            state.step = Step.CLOSED_FAILURE
            state.clear_card()  # wipe card data on cancel (good hygiene)
            return Responses.PAYMENT_CANCELLED
        # Unclear reply -> ask again, do NOT charge.
        return Responses.CONFIRM_UNCLEAR

    def _process_payment(self, state: SessionState) -> str:
        """
        Actually call the API to charge the card, then report the outcome:
          success -> recap with the transaction id, session done
          failure -> hand off to _handle_payment_error (retry or close)
        """
        card = state.card
        result = self._api.process_payment(   # <-- the real charge
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
        """
        The API rejected the payment. Decide what to do based on the error:
          - FIXABLE (bad card / bad amount / not enough balance) -> send the user
            back to fix that specific thing (up to a retry limit).
          - TERMINAL (or out of retries) -> close cleanly.
        The message + fixable/terminal decision come from PAYMENT_ERROR_GUIDANCE
        in responses.py.
        """
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
        """
        Used when the API is unreachable. Says "having trouble, try again" a few
        times; if it keeps failing, close the session cleanly rather than loop.
        """
        # Count the trouble; too many in a row -> give up gracefully.
        state.network_failures += 1
        if state.network_failures >= self._config.network_max_retries + 2:
            state.step = Step.CLOSED_FAILURE
            state.clear_card()
            return Responses.NETWORK_ERROR_TERMINAL
        return Responses.NETWORK_ERROR
