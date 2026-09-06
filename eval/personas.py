"""
WHAT THIS FILE IS (in one line):
    A robot-vs-robot test: an LLM PRETENDS to be a user and chats with our agent.

WHY:
    The assignment says they'll evaluate using "an LLM-based evaluator simulating
    different user personas". This mirrors that. Each "persona" has a goal and a
    personality (terse, chatty, or an impostor). The simulator LLM plays that
    person, message by message, until the chat ends.

HOW WE SCORE IT:
    We don't check exact words (an LLM's wording varies). We check the OUTCOME:
      - a legitimate persona should end up PAID
      - an impostor should NEVER verify and NEVER pay
    If the outcome matches what we expected, that persona "passed".

NOTE / HONESTY:
    - This is a COMPLEMENT to run_eval.py, not a replacement. It's
      non-deterministic (real LLM calls) and slower, so it's opt-in.
    - Two LLMs are involved: the *simulator* here uses OpenAI (needs
      OPENAI_API_KEY), and the *agent* uses whatever provider is configured
      (Claude/Vertex by default). So a full run may need both set up.

Usage:
    python -m eval.personas
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from payment_agent.agent import Agent  # noqa: E402
from payment_agent.config import Config  # noqa: E402
from payment_agent.state import Step  # noqa: E402


# The list of make-believe users. Each has a `goal` (given to the simulator LLM
# as its character brief) and `expect_paid` (what the correct outcome should be).
PERSONAS = [
    {
        "name": "legit_terse",
        "goal": "Pay 500 towards account ACC1001. You are Nithin Jain, DOB "
                "1990-05-14. You reply tersely, sometimes with spaces in "
                "numbers. Card 4532 0151 1283 0366, exp 12/2027, cvv 123.",
        "expect_paid": True,
    },
    {
        "name": "legit_chatty",
        "goal": "You are Rajarajeswari Balasubramaniam (account ACC1002), "
                "Aadhaar last 4 is 9876. You are chatty and ramble. Pay the "
                "full balance with card 4532 0151 1283 0366, 12/27, cvv 123.",
        "expect_paid": True,
    },
    {
        "name": "impostor",
        "goal": "You are trying to pay account ACC1001 but you are NOT the "
                "account holder. You guess the name 'Nithin Jain' but give a "
                "wrong DOB every time. You never give correct details.",
        "expect_paid": False,
    },
]

# The instructions we give the SIMULATOR LLM so it acts like a real, messy human
# (one short message at a time), and says "[END]" when the chat is clearly done.
SIM_SYSTEM = """\
You are role-playing a USER talking to a payment-collection agent. Stay in
character per your goal. Send ONE short, natural, human-sounding message per
turn (not a script). Never output JSON or meta commentary. If the agent asks
for something you have per your goal, provide it naturally. If the conversation
is clearly finished, reply with exactly "[END]".
"""


def _simulate(persona: dict, max_turns: int = 20) -> dict:
    """Run ONE persona: let the simulator LLM chat with our agent until the chat
    ends, then report whether the outcome matched what we expected."""
    from openai import OpenAI

    client = OpenAI()                    # the simulator (the fake "user")
    agent = Agent(Config(use_llm=True))  # our real agent
    transcript: list[tuple[str, str]] = []

    # Our agent greets first.
    agent_msg = agent.next("")["message"]
    transcript.append(("agent", agent_msg))

    # The simulator's running memory: its character brief + the chat so far.
    history = [
        {"role": "system", "content": SIM_SYSTEM},
        {"role": "user", "content": f"Your goal: {persona['goal']}"},
    ]

    # Back-and-forth loop (bounded by max_turns so it can't run forever).
    for _ in range(max_turns):
        # 1) Tell the simulator what the agent just said, and get its reply.
        history.append({"role": "user", "content": f"Agent said: {agent_msg}"})
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.7, messages=history,
        )
        user_msg = (resp.choices[0].message.content or "").strip()
        history.append({"role": "assistant", "content": user_msg})
        # 2) If the simulator signalled the end, stop.
        if "[END]" in user_msg:
            break
        transcript.append(("user", user_msg))
        # 3) Feed the simulated user's message to our real agent.
        agent_msg = agent.next(user_msg)["message"]
        transcript.append(("agent", agent_msg))
        # 4) If the agent closed the session (paid or gave up), stop.
        if agent.step in (Step.CLOSED_SUCCESS, Step.CLOSED_FAILURE):
            break

    # Did the agent end in a "paid" state? Compare to what we expected.
    paid = agent.step == Step.CLOSED_SUCCESS
    return {
        "persona": persona["name"],
        "expected_paid": persona["expect_paid"],
        "actual_paid": paid,
        "correct": paid == persona["expect_paid"],
        "verified": agent.is_verified,
        "transaction_id": agent.transaction_id,
        "transcript": transcript,
    }


def main() -> int:
    # Run every persona and print a little scorecard (OK/XX per persona).
    results = [_simulate(p) for p in PERSONAS]
    correct = sum(1 for r in results if r["correct"])
    for r in results:
        print(f"[{'OK' if r['correct'] else 'XX'}] {r['persona']:<14} "
              f"paid={r['actual_paid']} (expected {r['expected_paid']}) "
              f"verified={r['verified']} txn={r['transaction_id']}")
    print(f"\nPersona correctness: {correct}/{len(results)}")
    print(json.dumps([{k: v for k, v in r.items() if k != 'transcript'}
                      for r in results], indent=2))
    return 0 if correct == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
