"""Central configuration for the payment agent.

All tunables live here so behaviour is easy to audit and adjust. Values can be
overridden via environment variables to keep secrets and deployment specifics
out of the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Load a local .env file (if present) so secrets like OPENAI_API_KEY can live
# outside the code and outside version control. This is best-effort: if
# python-dotenv isn't installed, we silently rely on real environment variables,
# so the app never breaks for anyone who didn't install it.
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
    """Parse a boolean environment variable consistently."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration.

    Frozen so it cannot be mutated mid-conversation, which keeps behaviour
    deterministic across the lifetime of an Agent instance.
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
    llm_model: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o-mini")
    )
    llm_api_key: str | None = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )
    llm_timeout_seconds: float = field(
        default_factory=lambda: float(os.environ.get("LLM_TIMEOUT", "15"))
    )
    # Temperature 0 => as deterministic as the model allows. We want stable
    # extraction, not creativity.
    llm_temperature: float = 0.0


# A module-level default that most callers can share.
DEFAULT_CONFIG = Config()
