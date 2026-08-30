"""The public Agent class - the exact interface required by the evaluator.

    agent = Agent()
    agent.next("Hi")  # -> {"message": "..."}

The Agent is a thin facade: it owns the session state, the extractor, and the
orchestrator, and wires one user turn through them. All conversation state is
held internally between calls; no external setup is needed between turns.
"""

from __future__ import annotations

import logging

from .config import Config, DEFAULT_CONFIG
from .extractors import build_extractor
from .extractors.base import Expecting, Extractor
from .metrics import Metrics
from .orchestrator import Orchestrator
from .state import SessionState, Step
from .tools.payment_api import PaymentApiClient

logger = logging.getLogger("payment_agent.agent")


class Agent:
    """Conversational payment-collection agent.

    One `Agent` instance == one conversation. Construct a fresh `Agent()` to
    start a new session.
    """

    def __init__(
        self,
        config: Config = DEFAULT_CONFIG,
        *,
        api_client: PaymentApiClient | None = None,
        extractor: Extractor | None = None,
    ):
        self._config = config
        self._state = SessionState()
        self.metrics = Metrics()  # in-memory observability for this conversation
        # Give the API client our metrics collector (unless a client was injected).
        self._api = api_client or PaymentApiClient(config, metrics=self.metrics)
        self._extractor = extractor or build_extractor(config)
        self._orchestrator = Orchestrator(self._api, config, metrics=self.metrics)

    def next(self, user_input: str) -> dict:
        """Process exactly one turn and return {"message": str}."""
        self.metrics.incr("turns")
        with self.metrics.timer("turn"):
            return {"message": self._run_turn(user_input)}

    def _run_turn(self, user_input: str) -> str:
        # Very first turn with no meaningful input -> greet.
        text = (user_input or "").strip()
        if self._state.step == Step.GREETING and not self._has_actionable(text):
            # Greet and move to awaiting the account id.
            self._state.step = Step.AWAIT_ACCOUNT
            from .responses import Responses

            return Responses.GREETING

        expecting = self._orchestrator.expecting_for(self._state)
        try:
            with self.metrics.timer("extraction"):
                extracted = self._extractor.extract(text, expecting)
                # Out-of-order handling: while collecting the account id, the
                # user may also volunteer identity details. Run a second
                # identity pass and merge so we capture that data early.
                if expecting == Expecting.ACCOUNT:
                    identity_pass = self._extractor.extract(text, Expecting.IDENTITY)
                    extracted.merge_missing_from(identity_pass)
            # Observe whether the LLM ran or we fell back to rule-based.
            self.metrics.incr(f"extraction.source.{extracted.source}")
            if "llm_failed" in extracted.notes:
                self.metrics.incr("extraction.llm_failed")
        except Exception:  # extraction must never crash a turn
            logger.exception("Extractor raised; using empty extraction.")
            from .extractors.base import ExtractionResult

            extracted = ExtractionResult()
            self.metrics.incr("extraction.crashed")

        return self._orchestrator.handle(self._state, extracted)

    # --- helpers ----------------------------------------------------------- #
    def _has_actionable(self, text: str) -> bool:
        """Does the first message already carry something worth processing
        (e.g. the account id)? If so we skip the pure greeting and act."""
        if not text:
            return False
        from .extractors.base import Expecting

        try:
            ex = self._extractor.extract(text, Expecting.ACCOUNT)
            return bool(ex.account_id)
        except Exception:
            return False

    # Convenience accessors for tests / eval (read-only view of state).
    @property
    def step(self) -> Step:
        return self._state.step

    @property
    def is_verified(self) -> bool:
        return self._state.is_verified

    @property
    def transaction_id(self) -> str | None:
        return self._state.transaction_id
