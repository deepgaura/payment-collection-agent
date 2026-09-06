# Evaluation Approach

How this agent is tested, what "correct" means, the scripts that run it, and
where it struggles. Two complementary layers: a deterministic scenario harness
(the canonical, CI-gating eval) and a live LLM persona simulator.

## Test cases

The scenario harness (`eval/scenarios.py`) defines **15 scenarios** covering all
four required categories:

**Happy path (2)**
- `happy_path_messy_inputs` — every field given in messy natural language.
- `happy_path_full_balance` — "clear the full amount" resolves to the balance.

**Verification failure (2)**
- `verification_failure_exhausted` — wrong factor 3×, clean lockout.
- `verification_wrong_name` — correct factor but wrong name is rejected.

**Payment failure (5)**
- `payment_invalid_card_local` — Luhn failure caught locally, never charged.
- `payment_expired_card_local` — expired card caught locally, never charged.
- `payment_insufficient_balance_server` — server 422, routed back to fix amount.
- `payment_invalid_amount_server` — server 422, communicated clearly.
- `payment_declined_at_confirmation` — user says no at the confirmation step;
  cancelled cleanly, card never charged.

**Edge cases (6)**
- `edge_leap_year_exact` — 1988-02-29 verifies.
- `edge_leap_year_nearby_wrong` — 1988-02-28 fails strict match.
- `edge_zero_balance_nothing_to_pay` — ACC1003 (₹0.00) closes cleanly.
- `edge_out_of_order` — name volunteered with the account id.
- `edge_account_not_found_recovery` — bad id, then a valid one.
- `edge_early_payment_blocked` — card volunteered pre-verification is not acted on.

## How correctness is measured (per step)

"Correct" is defined concretely per step, and checked against both the message
text and the agent's observable state plus the mock API's recorded calls:

- **Account lookup** — the correct account is fetched; lookup called the right
  number of times.
- **Verification** — succeeds **only** when the strict rule holds (exact name +
  one matching factor); retries counted; stored data never leaked.
- **Amount** — the parsed amount matches intent and respects the balance.
- **Payment (tool call)** — the payment API is called with a correct payload
  **at the right time**, and — via negative assertions — **not** called before
  verification or when input fails local validation.
- **Outcome / closure** — success surfaces a transaction id; failures give a
  clear reason; the appropriate terminal state is reached.

**Metrics reported** by the harness: scenario success rate, per-category rate,
check-level accuracy, and **tool-call correctness as its own metric** (the doc
names "correctness of tool calls"), including the negative assertions above.

## Automated evaluation scripts

**1. Scenario harness — `python -m eval.run_eval [--verbose] [--live] [--llm]`**
Deterministic; uses a mock API so tool calls can be asserted precisely. Latest:

```
Scenario success rate   : 15/15 (100.0%)
Check-level accuracy     : 85/85 (100.0%)
Tool-call correctness    : 20/20 (100.0%)
```
Unit + flow tests (`python -m pytest`): **92 passing**.

**2. LLM persona simulator — `python -m eval.personas`**
Mirrors the stated evaluation method: an LLM role-plays users (terse legit,
chatty legit, impostor) and talks to the agent turn-by-turn. Correctness is
judged structurally — legit personas must reach a paid state; the impostor must
never verify and never pay. Verified against a live model + the real payment
API: **3/3 personas correct** (both legit personas paid with real transaction
IDs; the impostor was rejected and never charged).

## Observations — where the agent struggles

(From actually running both harnesses.)

- **LLM latency & non-determinism.** With the LLM on, each turn makes a model
  call (noticeable latency) and wording varies run to run. This is why the
  scenario harness uses the deterministic extractor as the canonical, CI-gating
  eval, and the persona sim is a complementary, non-gating check.
- **Rule-based fallback misses very unusual phrasing.** With the LLM off, the
  fallback can fail to extract exotic wording — but it fails *safe* (asks for
  clarification) rather than acting wrongly.
- **Balance is read once.** The server doesn't persist balance, so the recap
  computes the remaining balance for the session (original − paid) rather than
  re-querying.
- **Ambiguity handling is general, not per-field.** A turn contributing nothing
  usable for the current step gets a clear "I didn't catch that, here's what I
  need" instead of a silent loop; genuinely unclear numbers prompt a
  clarification without burning a retry.
- **Amount edge with the LLM.** Wording like "a couple hundred" is
  interpretation-dependent; the amount is always echoed back ("Got it -
  ₹200.00 …") so a mis-parse is correctable before any charge.
