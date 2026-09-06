"""
WHAT THIS FILE IS (in one line):
    A universal remote for talking to an LLM - it hides WHICH LLM you're using.

THE PROBLEM IT SOLVES:
    We might use Anthropic's Claude (on Google Vertex) OR OpenAI's GPT. Each has
    a slightly different way of being called. Instead of scattering "if openai
    do this, if claude do that" all over the code, we put it ALL in this one
    file. The rest of the code just says "give me an answer" and doesn't care
    which model is behind it. Switching models = change a setting, not the code.

TWO WAYS TO ASK THE LLM SOMETHING:
    - complete_json(...)  -> "answer me in strict JSON"  (used to extract fields)
    - complete_text(...)  -> "answer me in plain text"   (used to reword replies)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from .config import Config

logger = logging.getLogger("payment_agent.llm")


class LLMClient:
    """The universal remote. You build it once; it knows which model to call."""

    def __init__(self, config: Config):
        # Read the settings and set up the correct client for the chosen model.
        self._config = config
        self._provider = config.llm_provider   # "vertex" or "openai"
        self._client = None

        if self._provider == "vertex":
            # Claude, running on Google Vertex. Uses your gcloud login for auth.
            from anthropic import AnthropicVertex

            self._client = AnthropicVertex(
                region=config.vertex_region,
                project_id=config.vertex_project_id,
            )
        elif self._provider == "openai":
            # OpenAI's GPT. Uses an API key.
            from openai import OpenAI

            self._client = OpenAI(
                api_key=config.llm_api_key, timeout=config.llm_timeout_seconds
            )
        else:
            # Somebody set an unknown provider name in the config.
            raise ValueError(f"Unknown LLM provider: {self._provider}")

    # ----------------------------------------------------------------------- #
    # PUBLIC: the two things the rest of the code calls
    # ----------------------------------------------------------------------- #
    def complete_json(self, system: str, user: str) -> dict:
        """
        Ask the LLM and get back a PYTHON DICTIONARY (parsed from JSON).
        Used by the extractor, e.g. turning "acc 1001" into {"account_id": "ACC1001"}.
        `system` = the rulebook, `user` = the actual message to process.
        """
        text = self._complete(system, user, force_json=True)
        return json.loads(text)   # turn the JSON text into a real dict

    def complete_text(self, system: str, user: str, max_tokens: int = 160) -> str:
        """
        Ask the LLM and get back PLAIN TEXT.
        Used by the phraser to reword a reply in a friendly tone.
        """
        return self._complete(system, user, force_json=False, max_tokens=max_tokens).strip()

    # ----------------------------------------------------------------------- #
    # INTERNAL: the actual model calls (the provider-specific bits)
    # ----------------------------------------------------------------------- #
    def _complete(self, system: str, user: str, *, force_json: bool, max_tokens: int = 400) -> str:
        # Route to the right provider's method. `force_json=True` means
        # "make the model reply in JSON".
        if self._provider == "vertex":
            return self._complete_vertex(system, user, force_json, max_tokens)
        return self._complete_openai(system, user, force_json, max_tokens)

    def _complete_vertex(self, system, user, force_json, max_tokens) -> str:
        # --- How Claude wants to be called ---
        # The conversation is a list of messages. We add the user's message.
        messages = [{"role": "user", "content": user}]

        # TRICK for JSON: we make Claude's reply START with "{" by putting an
        # opening brace as the beginning of its answer. This nudges it to output
        # only JSON (no "Sure, here's the JSON:" chatter).
        if force_json:
            messages.append({"role": "assistant", "content": "{"})

        kwargs = dict(
            model=self._config.llm_model,
            max_tokens=max_tokens,
            temperature=self._config.llm_temperature,  # 0 = most consistent
            system=system,                              # the rulebook
            messages=messages,
        )
        try:
            resp = self._client.messages.create(**kwargs)
        except TypeError:
            # Some older versions of the Claude library don't accept every
            # setting (e.g. `temperature`). Rather than crash, drop that setting
            # and try once more.
            kwargs.pop("temperature", None)
            resp = self._client.messages.create(**kwargs)

        text = resp.content[0].text
        # Since we forced the reply to start with "{", we glue that "{" back on
        # to make the JSON complete again.
        return ("{" + text) if force_json else text

    def _complete_openai(self, system, user, force_json, max_tokens) -> str:
        # --- How OpenAI wants to be called ---
        kwargs = {}
        if force_json:
            # OpenAI has a built-in "reply in JSON" switch.
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(
            model=self._config.llm_model,
            temperature=self._config.llm_temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},   # the rulebook
                {"role": "user", "content": user},        # the message to handle
            ],
            **kwargs,
        )
        # Return the text; if it's somehow empty, return "{}" for JSON so the
        # caller's json.loads doesn't blow up.
        return resp.choices[0].message.content or ("{}" if force_json else "")


def build_llm_client(config: Config) -> Optional[LLMClient]:
    """
    Safely try to build an LLMClient.
      - If the LLM isn't configured/available -> return None (feature just off).
      - If building it fails for any reason    -> log a warning, return None.
    It NEVER raises, so the agent always keeps running (falling back to the
    deterministic path when there's no client).
    """
    if not config.llm_available:
        return None
    try:
        return LLMClient(config)
    except Exception as exc:
        logger.warning("Could not build LLM client (%s); LLM features off.", exc.__class__.__name__)
        return None
