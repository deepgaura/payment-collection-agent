"""Evaluation harness.

Runs every scenario in `scenarios.py` against a fresh Agent (wired to the
deterministic mock API so results are reproducible and free) and reports:

  - per-scenario pass/fail with the first failing check
  - per-category success rate
  - overall check-level accuracy (fraction of individual assertions passed)
  - tool-call correctness (were lookups/payments made correctly and only when
    appropriate)

Usage:
    python -m eval.run_eval            # offline, deterministic (recommended)
    python -m eval.run_eval --live     # run against the real API
    python -m eval.run_eval --llm      # use the LLM extractor (needs API key)

The offline mode is the canonical evaluation: it is deterministic and asserts
tool-call correctness precisely via the mock's recorded calls.
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
    name: str
    category: str
    passed: bool
    total_checks: int
    passed_checks: int
    failures: list[str]
    tool_total: int = 0     # tool-call-correctness checks in this scenario
    tool_passed: int = 0


def _build_agent(scenario, live: bool, use_llm: bool):
    """Return (agent, api) for one scenario run."""
    config = Config(use_llm=use_llm)
    if live:
        agent = Agent(config)
        return agent, agent._api  # real client; payment_calls not tracked
    # Offline: mock API + deterministic rule-based extractor unless --llm.
    from tests.fixtures import MockApiClient
    from payment_agent.extractors.rule_based import RuleBasedExtractor

    # A scenario may supply a custom API client (e.g. to force a server error).
    api = scenario.api_factory() if scenario.api_factory else MockApiClient()
    extractor = None if use_llm else RuleBasedExtractor()
    agent = Agent(config, api_client=api, extractor=extractor)
    return agent, api


def run_scenario(scenario: Scenario, *, live: bool, use_llm: bool) -> ScenarioResult:
    agent, api = _build_agent(scenario, live, use_llm)
    total = passed = 0
    tool_total = tool_passed = 0
    failures: list[str] = []

    def record(check, message, where):
        nonlocal total, passed, tool_total, tool_passed
        ok, label = check(message, agent, api)
        total += 1
        is_tool = getattr(check, "is_tool_check", False)
        if is_tool:
            tool_total += 1
        if ok:
            passed += 1
            if is_tool:
                tool_passed += 1
        else:
            failures.append(f"{where}: FAILED {label} | got: {message!r}")

    for i, turn in enumerate(scenario.turns):
        message = agent.next(turn.user)["message"]
        for check in turn.checks:
            record(check, message, f"turn {i} ({turn.user!r})")

    for check in scenario.final_checks:
        record(check, "", "final")

    return ScenarioResult(
        name=scenario.name, category=scenario.category,
        passed=(passed == total), total_checks=total,
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
    parser = argparse.ArgumentParser(description="Payment agent evaluation harness")
    parser.add_argument("--live", action="store_true", help="run against the real API")
    parser.add_argument("--llm", action="store_true", help="use the LLM extractor")
    parser.add_argument("--verbose", action="store_true", help="print failing checks")
    args = parser.parse_args()

    if args.live and any(s.final_checks for s in all_scenarios()):
        print("NOTE: --live cannot assert tool-call counts (no mock). "
              "Behavioural checks still run.\n")

    results = [run_scenario(s, live=args.live, use_llm=args.llm) for s in all_scenarios()]

    # --- report ---
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
