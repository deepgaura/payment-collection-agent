"""
WHAT THIS FILE IS (in one line):
    The test-runner for the whole agent. It plays scripted conversations and
    prints a scorecard.

HOW IT WORKS:
    - scenarios.py has a list of scripted chats (each = a list of user messages
      plus "checks" that say what SHOULD happen).
    - This file runs each scenario against a fresh Agent, ticks off the checks,
      and prints a report: which scenarios passed, and some scores.

WHAT IT REPORTS:
    - per-scenario PASS/FAIL
    - success rate per category (happy / verification_failure / payment_failure / edge)
    - check-level accuracy (fraction of ALL individual checks that passed)
    - tool-call correctness (were the lookup/payment APIs called correctly and
      only when they should be)

HOW TO RUN:
    python -m eval.run_eval            # normal: offline + a fake API (fast, repeatable)
    python -m eval.run_eval --live     # use the REAL payment API
    python -m eval.run_eval --llm      # use the LLM translator (needs a key)
    add --verbose                      # also print which checks failed

    "Offline" mode is the main one - it's repeatable and can check tool calls
    exactly (because the fake API records every call).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass

# Make src/ importable when run from the repo root.
sys.path.insert(0, "src")

from payment_agent.agent import Agent  # noqa: E402
from payment_agent.config import Config  # noqa: E402

from .scenarios import Scenario, all_scenarios  # noqa: E402


@dataclass
class ScenarioResult:
    """The scorecard for ONE scenario after we run it."""
    name: str
    category: str
    passed: bool             # did ALL checks in this scenario pass?
    total_checks: int        # how many checks it had
    passed_checks: int       # how many passed
    failures: list[str]      # descriptions of the ones that failed
    tool_total: int = 0      # how many of the checks were tool-call checks
    tool_passed: int = 0     # how many of those passed


def _build_agent(scenario, live: bool, use_llm: bool):
    """
    Set up a fresh Agent for one scenario, wired the way this run wants:
      live=True  -> talk to the REAL API
      live=False -> use a FAKE API (records calls so we can check tool usage)
    Returns (agent, api) - we return the api too so checks can inspect it.
    """
    config = Config(use_llm=use_llm)
    if live:
        agent = Agent(config)
        return agent, agent._api  # real client; it doesn't record calls
    # Offline: use the fake API + the regex translator (unless --llm was passed).
    from tests.fixtures import MockApiClient
    from payment_agent.extractors.rule_based import RuleBasedExtractor

    # Some scenarios bring their own fake API (e.g. to force "insufficient balance").
    api = scenario.api_factory() if scenario.api_factory else MockApiClient()
    extractor = None if use_llm else RuleBasedExtractor()
    agent = Agent(config, api_client=api, extractor=extractor)
    return agent, api


def run_scenario(scenario: Scenario, *, live: bool, use_llm: bool) -> ScenarioResult:
    """Play one scripted chat and tick off every check, returning its scorecard."""
    agent, api = _build_agent(scenario, live, use_llm)
    total = passed = 0
    tool_total = tool_passed = 0
    failures: list[str] = []

    # Small helper: run one check and update the tallies.
    def record(check, message, where):
        nonlocal total, passed, tool_total, tool_passed
        ok, label = check(message, agent, api)     # each check returns (passed?, label)
        total += 1
        is_tool = getattr(check, "is_tool_check", False)   # is it a tool-call check?
        if is_tool:
            tool_total += 1
        if ok:
            passed += 1
            if is_tool:
                tool_passed += 1
        else:
            failures.append(f"{where}: FAILED {label} | got: {message!r}")

    # Play each user turn, then run that turn's checks against the agent's reply.
    for i, turn in enumerate(scenario.turns):
        message = agent.next(turn.user)["message"]
        for check in turn.checks:
            record(check, message, f"turn {i} ({turn.user!r})")

    # Finally, run the "end-of-conversation" checks (e.g. "did we end paid?").
    for check in scenario.final_checks:
        record(check, "", "final")

    return ScenarioResult(
        name=scenario.name, category=scenario.category,
        passed=(passed == total), total_checks=total,   # scenario passes only if EVERY check did
        passed_checks=passed, failures=failures,
        tool_total=tool_total, tool_passed=tool_passed,
    )


def main() -> int:
    # Windows consoles default to cp1252 and choke on the rupee sign in verbose
    # failure output; force UTF-8 so the report always prints.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    # Read the command-line flags (--live / --llm / --verbose).
    parser = argparse.ArgumentParser(description="Payment agent evaluation harness")
    parser.add_argument("--live", action="store_true", help="run against the real API")
    parser.add_argument("--llm", action="store_true", help="use the LLM extractor")
    parser.add_argument("--verbose", action="store_true", help="print failing checks")
    args = parser.parse_args()

    if args.live and any(s.final_checks for s in all_scenarios()):
        print("NOTE: --live cannot assert tool-call counts (no mock). "
              "Behavioural checks still run.\n")

    # Run every scenario and collect its scorecard.
    results = [run_scenario(s, live=args.live, use_llm=args.llm) for s in all_scenarios()]

    # --- print the report ---
    # Group scenarios by category so we can show a per-category success rate.
    by_category: dict[str, list[ScenarioResult]] = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)

    print("=" * 70)
    print("PAYMENT AGENT EVALUATION")
    mode = "LIVE" if args.live else "OFFLINE(mock)"
    nlu = "LLM" if args.llm else "rule-based"
    print(f"mode={mode}  nlu={nlu}  scenarios={len(results)}")
    print("=" * 70)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name:<38} {r.passed_checks}/{r.total_checks} checks")
        if not r.passed and args.verbose:
            for f in r.failures:
                print(f"        - {f}")

    print("-" * 70)
    print("Success rate by category:")
    for cat, rs in sorted(by_category.items()):
        n_pass = sum(1 for r in rs if r.passed)
        print(f"  {cat:<22} {n_pass}/{len(rs)} scenarios "
              f"({100 * n_pass / len(rs):.0f}%)")

    total_checks = sum(r.total_checks for r in results)
    passed_checks = sum(r.passed_checks for r in results)
    scenarios_passed = sum(1 for r in results if r.passed)
    tool_total = sum(r.tool_total for r in results)
    tool_passed = sum(r.tool_passed for r in results)

    print("-" * 70)
    print("METRICS")
    print(f"  Scenario success rate   : {scenarios_passed}/{len(results)} "
          f"({100 * scenarios_passed / len(results):.1f}%)")
    print(f"  Check-level accuracy    : {passed_checks}/{total_checks} "
          f"({100 * passed_checks / total_checks:.1f}%)")
    # The doc explicitly names "correctness of tool calls" as a metric, so we
    # surface it on its own: were the lookup/payment APIs called the right number
    # of times, with the right amount, and never when they shouldn't be.
    if tool_total:
        print(f"  Tool-call correctness   : {tool_passed}/{tool_total} "
              f"({100 * tool_passed / tool_total:.1f}%)")
    print("=" * 70)

    return 0 if scenarios_passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
