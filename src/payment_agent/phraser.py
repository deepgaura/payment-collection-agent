"""
WHAT THIS FILE IS (in one line):
    It takes the agent's plain, robotic reply and makes it sound friendly/human.

THINK OF IT LIKE THIS:
    The "brain" (orchestrator) decides WHAT to say, e.g.:
        "How much would you like to pay?"
    This file (the "voice") rewords it to sound warm, e.g.:
        "Sure! How much would you like to pay today?"
    The MEANING never changes - only the tone.

WHY IT'S SAFE (the important part):
    The LLM here is ONLY allowed to reword harmless messages (greetings, "what's
    your card number", etc.). It is NEVER allowed to touch sensitive messages
    like the balance, the transaction id, or verification results - those are
    shown exactly as the code wrote them. So the LLM can't leak or make up money
    details. And if the LLM ever errors, we just show the original plain message.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("payment_agent.phraser")


# ---------------------------------------------------------------------------
# WHICH MESSAGES ARE ALLOWED TO BE REWORDED?
# ---------------------------------------------------------------------------
# Only these "intents" (types of message) may be made friendly by the LLM.
# Each message the brain produces gets tagged with one of these labels.
#
#   "greeting"    -> "Hello! Share your account ID."
#   "ask_account" -> "I didn't catch an account ID, could you share it?"
#   "ask_amount"  -> "How much would you like to pay?"
#   "ask_card"    -> "What's your card number?"
#   "clarify"     -> "Sorry, I didn't catch that."
#
# Anything NOT in this list (balance, transaction id, verification result,
# payment errors, the confirmation summary) is shown EXACTLY as written - the
# LLM never sees it.
#
# WHY is "ask for name / date-of-birth" NOT in this list?
#   Because if the user typed a wrong name and we reworded the reply to
#   "Got it, Nithin!", it would SOUND like we accepted the name - but we
#   haven't checked it yet. To avoid that false impression, identity questions
#   are always shown as the plain, exact template.
SAFE_INTENTS = frozenset({
    "greeting",
    "ask_account",
    "ask_amount",
    "ask_card",
    "clarify",
})


# ---------------------------------------------------------------------------
# THE INSTRUCTIONS WE GIVE THE LLM (the "system prompt")
# ---------------------------------------------------------------------------
# This is the rulebook the LLM must follow when rewording. In plain words it
# says: "Make this sound nice, but do NOT change any facts, do NOT make things
# up, and do NOT pretend anything was accepted."
_SYSTEM = """\
You are the *voice* of a payment-collection assistant. You do NOT make decisions
and you do NOT know any account details. You are given the assistant's intended
reply (already decided by the system) and, optionally, the user's last message.

Your only job: rewrite the intended reply so it sounds warm, natural, and human,
optionally acknowledging the user's aside in ONE short clause, then delivering
the same request/information.

HARD RULES:
- Preserve the meaning exactly. Do NOT add, remove, or change any facts, numbers,
  amounts, names, or requirements.
- Do NOT invent details, make promises, or ask for anything the intended reply
  didn't ask for.
- Do NOT imply anything has been accepted, confirmed, matched, or verified unless
  the intended reply explicitly says so. Never affirm a name or detail the user
  just gave (no "Got it, <name>!" style acceptance).
- Keep it to 1-2 short sentences. Friendly and professional, not chatty or salesy.
- No emojis. Output ONLY the rewritten reply text, nothing else.
"""


class Phraser:
    """The little helper that (optionally) rewords replies to sound friendly."""

    def __init__(self, client, metrics=None):
        # `client` = the thing that actually talks to the LLM (Claude/OpenAI).
        #            If it's None, this whole feature is simply OFF.
        # `metrics` = optional counter, just to record how often we reword.
        self._client = client
        self._metrics = metrics

    def enabled(self) -> bool:
        # "Is the friendly-rewording feature turned on?"
        # It's on only if we were given a working LLM client.
        return self._client is not None

    def phrase(self, base_message: str, intent: str, user_text: str) -> str:
        """
        Take the plain reply and return a friendlier version.

        Inputs:
          base_message = the exact reply the brain wrote
                         e.g. "How much would you like to pay?"
          intent       = the label for this message, e.g. "ask_amount"
          user_text    = what the user just typed, e.g. "ok cool"

        Returns: a reworded message, OR the original if we're not allowed to
                 reword this one (or if anything goes wrong).
        """

        # STEP 1: Bail out (return the original, untouched) if either:
        #   - the feature is off (no LLM client), OR
        #   - this message type is NOT in our safe list (e.g. it's the balance).
        # This is the safety gate that protects sensitive messages.
        if self._client is None or intent not in SAFE_INTENTS:
            return base_message

        try:
            # STEP 2: Build the note we hand to the LLM. We give it the user's
            # last message (for a natural acknowledgement) and the exact reply
            # we want reworded.
            user_prompt = (
                f"User's last message: {user_text!r}\n"
                f"Assistant's intended reply: {base_message!r}\n"
                "Rewrite the intended reply per your rules."
            )

            # (bookkeeping) count that we made a phrasing call.
            if self._metrics is not None:
                self._metrics.incr("phraser.calls")

            # STEP 3: Ask the LLM to reword it.
            out = self._client.complete_text(_SYSTEM, user_prompt, max_tokens=160)

            # STEP 4: Safety check on what came back. If the LLM returned
            # nothing / junk (empty or too short to be a real sentence), don't
            # use it - fall back to the original plain reply.
            if not out or len(out) < 3:
                return base_message

            # All good - return the friendly version.
            return out

        except Exception as exc:
            # STEP 5: If ANYTHING broke (LLM timeout, network, error), never
            # crash the conversation. Just quietly use the original reply.
            logger.warning("Phrasing failed (%s); using base message.", exc.__class__.__name__)
            if self._metrics is not None:
                self._metrics.incr("phraser.failed")
            return base_message
