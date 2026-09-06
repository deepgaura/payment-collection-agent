"""
WHAT THIS FILE IS (in one line):
    All the knobs/settings in one place (API URL, timeouts, retry limits, which
    LLM to use). Nothing here makes decisions - it just holds numbers and flags.

HOW SETTINGS ARE CHOSEN:
    Each setting has a sensible default, but can be overridden with an
    environment variable (often set in a local `.env` file). So you can change
    behaviour without touching code - e.g. set USE_LLM=false to run offline.

WHO READS IT:
    Basically everyone: the API client (URL/timeouts), the orchestrator (retry
    limits), agent.py (which LLM/phraser to build). One shared `DEFAULT_CONFIG`
    is created at the bottom.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Load a local ".env" file (if there is one) so secrets like the API key live
# outside the code (and outside git). If python-dotenv isn't installed, we just
# fall back to the real environment - the app still runs.
try:
    from dotenv import load_dotenv

    # override=True so the .env file is authoritative for local runs, even if a
    # stale shell variable of the same name is lingering in the environment.
    load_dotenv(override=True)
except Exception:
    pass


# --------------------------------------------------------------------------- #
# API configuration
# --------------------------------------------------------------------------- #
DEFAULT_BASE_URL = (
    "https://se-payment-verification-api.service.external.usea2.aws.prodigaltech.com"
)


def _env_bool(name: str, default: bool) -> bool:
    """Read a true/false setting from an env var. "true"/"1"/"yes"/"on" -> True."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    """
    All the settings, read once when the agent starts. It's "frozen" (can't be
    changed after creation) so behaviour stays consistent for the whole chat.
    Each field below reads an env var (with a default) - the comments explain
    what each one does.
    """

    # --- API client ---
    base_url: str = field(
        default_factory=lambda: os.environ.get("PAYMENT_API_BASE_URL", DEFAULT_BASE_URL)
    )
    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("PAYMENT_API_TIMEOUT", "20"))
    )
    # Number of automatic retries for *transient* failures (network / 5xx).
    # Business errors (404, 422) are never retried automatically.
    network_max_retries: int = field(
        default_factory=lambda: int(os.environ.get("PAYMENT_API_RETRIES", "2"))
    )
    network_retry_backoff_seconds: float = 0.5

    # --- Verification policy ---
    # Number of failed verification attempts allowed before the conversation
    # is terminated. "Reasonable retries with a sensible limit" per the spec.
    max_verification_attempts: int = field(
        default_factory=lambda: int(os.environ.get("MAX_VERIFICATION_ATTEMPTS", "3"))
    )

    # --- Liveness ---
    # Consecutive no-progress turns (user never supplies what's needed, keeps
    # chatting) before we close gracefully. Bounds the whole conversation so it
    # can't loop forever, distinct from the wrong-attempt limits above.
    max_no_progress_turns: int = field(
        default_factory=lambda: int(os.environ.get("MAX_NO_PROGRESS_TURNS", "5"))
    )

    # --- Payment policy ---
    # How many times the user may correct a fixable payment problem (bad card,
    # bad amount, insufficient balance) before we close the conversation.
    max_payment_attempts: int = field(
        default_factory=lambda: int(os.environ.get("MAX_PAYMENT_ATTEMPTS", "3"))
    )
    # How many times we re-ask for an account id that does not exist.
    max_account_lookup_attempts: int = field(
        default_factory=lambda: int(os.environ.get("MAX_ACCOUNT_LOOKUP_ATTEMPTS", "3"))
    )

    # --- LLM extraction layer ---
    # The LLM is the PRIMARY NLU/extraction layer and is ON by default - this is
    # an AI agent, and the LLM is what interprets messy natural language. When a
    # key is present it is used; if there is no key (or the call fails), the
    # agent transparently falls back to the deterministic rule-based extractor,
    # so it always runs and never breaks a turn.
    #
    # On determinism: all flow, verification, validation, and API decisions live
    # in deterministic code - only free-text *understanding* uses the LLM, at
    # temperature 0. The same input therefore drives the same flow decisions
    # across runs (verification passes/fails consistently, no random state
    # jumps); only the exact wording of extraction can vary, which never changes
    # whether verification or payment succeeds.
    use_llm: bool = field(default_factory=lambda: _env_bool("USE_LLM", True))

    # Provider selects the backend: "vertex" (Anthropic Claude on Google Vertex)
    # or "openai". Auto-detects: if Vertex env vars are present, default to
    # vertex; else openai. Both go through one small client abstraction.
    llm_provider: str = field(
        default_factory=lambda: os.environ.get(
            "LLM_PROVIDER",
            "vertex" if os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID") else "openai",
        ).lower()
    )
    llm_model: str = field(
        default_factory=lambda: os.environ.get(
            "LLM_MODEL",
            os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-5@20251101")
            if os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
            else "gpt-4o-mini",
        )
    )
    llm_api_key: str | None = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )
    # --- Vertex (Anthropic) settings ---
    vertex_project_id: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
    )
    vertex_region: str = field(
        default_factory=lambda: os.environ.get("CLOUD_ML_REGION", "europe-west1")
    )
    llm_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("LLM_TIMEOUT", "20"))
    )
    # Temperature 0 => as deterministic as the model allows. We want stable
    # extraction, not creativity.
    llm_temperature: float = 0.0

    # --- Conversational phrasing layer ---
    # When on, an LLM lightly *rephrases* the deterministically-chosen reply to
    # sound natural (acknowledge small talk, warmer prompts). It NEVER decides
    # anything and is only allowed on safe, non-transactional steps; sensitive
    # messages (balance, transaction id, errors) always use the exact
    # deterministic template. Falls back to the template on any error.
    use_conversational: bool = field(
        default_factory=lambda: _env_bool("USE_CONVERSATIONAL", True)
    )

    @property
    def llm_available(self) -> bool:
        """
        Can we actually use an LLM right now? Only if it's turned on AND we have
        what that provider needs (a Vertex project id, or an OpenAI key).
        agent.py checks this before building the LLM extractor/phraser; if it's
        False, the agent quietly uses the regex fallback instead.
        """
        if not self.use_llm:
            return False
        if self.llm_provider == "vertex":
            return bool(self.vertex_project_id)
        return bool(self.llm_api_key)


# One ready-made Config that most of the code shares by default.
DEFAULT_CONFIG = Config()
