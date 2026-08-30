"""Optional LLM-based persona simulator.

The assignment says evaluation is done by "an LLM-based evaluator ... simulating
different user personas". This module mirrors that: an LLM role-plays a user
with a goal and a personality (terse, chatty, types digits with spaces, etc.),
talking to our Agent in a loop until the conversation closes. It is a
complement to the deterministic scenario harness, not a replacement - it is
non-deterministic, so it is opt-in and requires an API key.

Correctness for a persona run is judged structurally (did a legitimate persona
reach a paid state? did an impostor persona fail verification and never pay?),
which does not depend on exact wording.

Usage:
    OPENAI_API_KEY=... python -m eval.personas
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from payment_agent.agent import Agent  # noqa: E402
from payment_agent.config import Config  # noqa: E402
from payment_agent.state import Step  # noqa: E402


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

SIM_SYSTEM = """\
You are role-playing a USER talking to a payment-collection agent. Stay in
character per your goal. Send ONE short, natural, human-sounding message per
turn (not a script). Never output JSON or meta commentary. If the agent asks
for something you have per your goal, provide it naturally. If the conversation
is clearly finished, reply with exactly "[END]".
"""


def _simulate(persona: dict, max_turns: int = 20) -> dict:
    from openai import OpenAI

    client = OpenAI()
    agent = Agent(Config(use_llm=True))
    transcript: list[tuple[str, str]] = []

    # Agent greets first.
    agent_msg = agent.next("")["message"]
    transcript.append(("agent", agent_msg))

    history = [
        {"role": "system", "content": SIM_SYSTEM},
        {"role": "user", "content": f"Your goal: {persona['goal']}"},
    ]

    for _ in range(max_turns):
        history.append({"role": "user", "content": f"Agent said: {agent_msg}"})
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.7, messages=history,
        )
        user_msg = (resp.choices[0].message.content or "").strip()
        history.append({"role": "assistant", "content": user_msg})
        if "[END]" in user_msg:
            break
        transcript.append(("user", user_msg))
        agent_msg = agent.next(user_msg)["message"]
        transcript.append(("agent", agent_msg))
        if agent.step in (Step.CLOSED_SUCCESS, Step.CLOSED_FAILURE):
            break

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
