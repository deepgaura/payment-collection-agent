"""
WHAT THIS FILE IS (in one line):
    The front door + coordinator. It's the `Agent` class the grader uses:
        agent = Agent()
        agent.next("Hi")   ->  {"message": "..."}

WHAT IT DOES:
    It doesn't make decisions itself. It just holds the pieces and wires ONE
    user turn through them, in this order:
        1. clean the text
        2. ask the EXTRACTOR to pull out fields   (this is the ONLY place
           extract() is called - the orchestrator never calls it)
        3. hand those fields to the ORCHESTRATOR (the brain) to decide the reply
        4. optionally soften the reply's tone via the PHRASER
        5. return {"message": ...}

    It also owns the "notebook" (SessionState), so everything is remembered
    between turns. One Agent object = one conversation.

THE PIECES IT HOLDS:
    self._state        -> the notebook (state.py)
    self._extractor    -> the translator (extractors/) - LLM or regex
    self._orchestrator -> the brain (orchestrator.py)
    self._phraser      -> optional friendly-tone layer (phraser.py)
    self._api          -> the API client (tools/payment_api.py)
    self.metrics       -> the stats notebook (metrics.py)
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
    """
    One `Agent` = one conversation. Make a new Agent() to start a fresh chat.
    """

    def __init__(
        self,
        config: Config = DEFAULT_CONFIG,
        *,
        # These two can be "injected" by tests (e.g. a fake API, a fixed
        # extractor). In normal use they're built automatically below.
        api_client: PaymentApiClient | None = None,
        extractor: Extractor | None = None,
    ):
        self._config = config
        self._state = SessionState()            # the blank notebook for this chat
        self.metrics = Metrics()                # stats for this conversation
        # API client (real one unless a test passed a fake). Shares our metrics.
        self._api = api_client or PaymentApiClient(config, metrics=self.metrics)
        # The translator: build_extractor picks the LLM one if available, else
        # the regex one. (See extractors/__init__.py.)
        self._extractor = extractor or build_extractor(config)
        # The brain. We give it the API client + metrics.
        self._orchestrator = Orchestrator(self._api, config, metrics=self.metrics)
        # Optional friendly-tone layer (None if disabled / no LLM).
        self._phraser = self._build_phraser(config)

    def next(self, user_input: str) -> dict:
        """
        THE PUBLIC METHOD the grader calls, once per user message.
        Always returns {"message": <text to show the user>}.
        (We time the whole turn for metrics, then delegate to _run_turn.)
        """
        self.metrics.incr("turns")
        with self.metrics.timer("turn"):
            return {"message": self._run_turn(user_input)}

    def _run_turn(self, user_input: str) -> str:
        # STEP 0: tidy the input ("  hi  " -> "hi"; None -> "").
        text = (user_input or "").strip()

        # STEP 1: if this is the very first message AND it has no account id
        # (e.g. just "hi"), simply greet and wait for the account id.
        if self._state.step == Step.GREETING and not self._has_actionable(text):
            self._state.step = Step.AWAIT_ACCOUNT
            from .responses import Responses
            return Responses.GREETING

        # STEP 2: ask the brain "what should we be extracting right now?"
        # (e.g. AMOUNT while collecting the amount) - this guides the translator.
        expecting = self._orchestrator.expecting_for(self._state)

        # STEP 3: run the TRANSLATOR (this is the ONLY place extract() is called).
        # We extract ONLY what the current step is asking for - one focused pass.
        try:
            with self.metrics.timer("extraction"):
                extracted = self._extractor.extract(text, expecting)
            # Record whether the LLM or the regex fallback produced this.
            self.metrics.incr(f"extraction.source.{extracted.source}")
            if "llm_failed" in extracted.notes:
                self.metrics.incr("extraction.llm_failed")
        except Exception:
            # A turn must NEVER crash. If the translator blew up, use an empty
            # result (the brain will then just re-ask for what it needs).
            logger.exception("Extractor raised; using empty extraction.")
            from .extractors.base import ExtractionResult
            extracted = ExtractionResult()
            self.metrics.incr("extraction.crashed")

        # STEP 4: hand the clean fields to the BRAIN, which decides the reply
        # and an "intent" label (how sensitive the reply is).
        message, intent = self._orchestrator.handle(self._state, extracted)

        # STEP 5: optionally make the reply sound friendlier. The phraser only
        # touches "safe" messages (never the balance/recap/errors) - see phraser.py.
        if self._phraser is not None and self._phraser.enabled():
            with self.metrics.timer("phrasing"):
                message = self._phraser.phrase(message, intent, text)
        return message

    def _build_phraser(self, config: Config):
        """
        Set up the friendly-tone layer - but only if it's turned on AND an LLM
        is actually available. If anything is missing, return None (the agent
        then just uses the plain template replies).
        """
        if not (config.use_conversational and config.llm_available):
            return None
        try:
            from .llm_client import build_llm_client
            from .phraser import Phraser

            client = build_llm_client(config)
            return Phraser(client, metrics=self.metrics) if client else None
        except Exception:
            return None

    # --- helpers ----------------------------------------------------------- #
    def _has_actionable(self, text: str) -> bool:
        """
        "Did the user's FIRST message already include an account id?"
        If yes -> skip the plain greeting and go straight to looking it up.
        Example: "pay my bill for ACC1001" -> True (don't waste a turn saying hi).
                 "hello there"              -> False (just greet).
        """
        if not text:
            return False
        try:
            ex = self._extractor.extract(text, Expecting.ACCOUNT)
            return bool(ex.account_id)
        except Exception:
            return False

    # --- read-only peek-holes into the notebook (used by tests / eval) ------ #
    # These just expose bits of state so tests can check progress. They don't
    # change anything.
    @property
    def step(self) -> Step:
        return self._state.step                 # which step are we on?

    @property
    def is_verified(self) -> bool:
        return self._state.is_verified          # has the user passed verification?

    @property
    def transaction_id(self) -> str | None:
        return self._state.transaction_id       # the txn id, once paid
