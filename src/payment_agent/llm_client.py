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
    def complete_json(self, system: str, user: str, schema: Optional[dict] = None) -> dict:
        """
        Ask the LLM and get back a PYTHON DICTIONARY (parsed from JSON).
        Used by the extractor, e.g. turning "acc 1001" into {"account_id": "ACC1001"}.
        `system` = the rulebook, `user` = the actual message to process.

        If `schema` (a JSON Schema dict) is given, we ask the PROVIDER to force
        the reply to match that exact shape - this is real "structured output":
          - OpenAI  -> response_format = json_schema (strict): the API guarantees
                       valid-against-schema JSON or errors out.
          - Claude  -> a single "tool" whose input_schema IS our schema, with
                       tool_choice forcing the model to call it; the tool's
                       arguments are the structured object.
        If the provider/library doesn't support it, we transparently fall back
        to plain JSON mode (still parsed the same way). The caller ALSO validates
        the result, so structure is never assumed from the model alone.
        """
        if schema is not None:
            data = self._complete_structured(system, user, schema)
            if data is not None:
                return data
            # Provider couldn't do schema mode -> fall through to plain JSON.
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

    # ----------------------------------------------------------------------- #
    # STRUCTURED OUTPUT: force the reply to match a JSON Schema at the PROVIDER.
    # Returns a parsed dict on success, or None if this provider/library build
    # can't do schema mode (so the caller falls back to plain JSON mode).
    # ----------------------------------------------------------------------- #
    def _complete_structured(self, system: str, user: str, schema: dict) -> Optional[dict]:
        try:
            if self._provider == "vertex":
                return self._structured_vertex(system, user, schema)
            return self._structured_openai(system, user, schema)
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            # The installed library/model doesn't support this shape of call.
            # Not an error worth crashing on - just fall back to plain JSON.
            logger.info(
                "Structured-output mode unavailable (%s); using plain JSON.",
                exc.__class__.__name__,
            )
            return None

    def _structured_openai(self, system: str, user: str, schema: dict) -> dict:
        # OpenAI's strict Structured Outputs: the API validates the model's
        # output against the schema and guarantees a conforming JSON object.
        from .extractors.schema import EXTRACTION_TOOL_NAME

        resp = self._client.chat.completions.create(
            model=self._config.llm_model,
            temperature=self._config.llm_temperature,
            max_tokens=400,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": EXTRACTION_TOOL_NAME,
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)

    def _structured_vertex(self, system: str, user: str, schema: dict) -> dict:
        # Anthropic's structured-output path: expose a single "tool" whose
        # input_schema is our schema, and FORCE the model to call it. The tool's
        # arguments (tool_use.input) are our structured object.
        from .extractors.schema import (
            EXTRACTION_TOOL_NAME,
            EXTRACTION_TOOL_DESCRIPTION,
        )

        kwargs = dict(
            model=self._config.llm_model,
            max_tokens=400,
            temperature=self._config.llm_temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": EXTRACTION_TOOL_NAME,
                    "description": EXTRACTION_TOOL_DESCRIPTION,
                    "input_schema": schema,
                }
            ],
            # Force the model to answer BY calling our tool (no free-form text).
            tool_choice={"type": "tool", "name": EXTRACTION_TOOL_NAME},
        )
        try:
            resp = self._client.messages.create(**kwargs)
        except TypeError:
            # Older client that doesn't accept `temperature` alongside tools.
            kwargs.pop("temperature", None)
            resp = self._client.messages.create(**kwargs)

        # Find the tool_use block and return its already-parsed input dict.
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        # Model didn't call the tool (shouldn't happen with forced tool_choice).
        raise ValueError("model did not return a tool_use block")

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
