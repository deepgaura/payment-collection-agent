"""Extraction (natural-language understanding) layer.

Turns messy free-form user text into structured candidate fields. Two
interchangeable implementations sit behind the `Extractor` interface:

- `LLMExtractor`  : primary, on by default. Uses an LLM with a strict
                    structured-output contract to interpret messy phrasing;
                    falls back to rule-based on any error.
- `RuleBasedExtractor` : deterministic fallback (no key / LLM error) and the
                    path used for reproducible tests and CI.
"""

from .base import Extractor, ExtractionResult
from .rule_based import RuleBasedExtractor

__all__ = ["Extractor", "ExtractionResult", "RuleBasedExtractor", "build_extractor"]


def build_extractor(config=None):
    """Factory: return the LLM extractor when enabled and importable, else the
    deterministic rule-based extractor. Falling back never raises."""
    from ..config import DEFAULT_CONFIG

    config = config or DEFAULT_CONFIG
    fallback = RuleBasedExtractor()
    if not config.llm_available:
        return fallback
    try:
        from .llm import LLMExtractor

        return LLMExtractor(config=config, fallback=fallback)
    except Exception:
        # Any import/config problem => stay functional with the rule-based path.
        return fallback
