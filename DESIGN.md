# Design Document — Payment Collection AI Agent

*(Evaluation approach is a separate deliverable — see `EVALUATION.md`.)*

## 1. Architecture overview

Core principle: **the LLM understands, deterministic code decides.** The LLM
turns messy natural language into structured fields; everything that must be
correct, strict, or auditable — verification, validation, flow, API calls — is
plain Python.

```
        Agent.next(user_input) -> {"message": str}
                     │
   ┌─────────────────┼───────────────────────┐
   ▼                 ▼                        ▼
Extractor       Orchestrator             API client
(NLU layer)    (state machine)          + validators
LLM primary    owns every decision:     lookup-account
 + rule-based  next step, when to        process-payment
 fallback      call APIs, verify,
   ▲            retry, error routing
   │                 │
 raw text            ▼
              SessionState (in-memory: step, account, claim,
              verified flag, amount, card, retry counters)
                     │
                     ▼
              Response composer (safe, templated)
```

Components:
- **`Agent`** — implements the required `next()` interface; owns the session and
  wires one turn: extract → orchestrate → respond.
- **Extractor** (`extractors/`) — the only place the LLM is used. `LLMExtractor`
  (primary, on by default) converts a message + a context hint into structured
  candidate fields via a strict JSON prompt; `RuleBasedExtractor` (regex/date
  parsing) is the fallback and the reproducible path for CI.
- **Orchestrator** — deterministic state machine. Owns the flow, decides when to
  call each API, whether verification passed, and how every error is handled.
- **Verifier / Validators** — strict name + secondary-factor matching; Luhn,
  CVV, expiry, amount, and strict date parsing, all run before any API call.
- **API client** — normalises every response into an `ApiResult`; retries only
  transient (network/5xx) failures, never business errors.
- **Responses** — the single place user-facing text is built, so "never leak
  sensitive data" is auditable in one file.
- **Metrics** (`metrics.py`) — a small in-memory collector wired through the
  Agent, API client, and orchestrator. Records latency (turn / LLM / per API
  endpoint), tool-call counts + error codes, verification/payment outcomes, and
  LLM-vs-rule-based usage. Exposed as `agent.metrics`; printed by the CLI.

Flow is an explicit `Step` enum (`GREETING → AWAIT_ACCOUNT → AWAIT_IDENTITY →
AWAIT_AMOUNT → AWAIT_CARD → PROCESSING → CLOSED_*`). Payment steps are only
reachable after `is_verified` is set, so "no payment before verification" is
structurally guaranteed rather than merely checked.

## 2. Key decisions and why

**LLM for understanding, deterministic code for decisions.** An LLM deciding
whether identity matches or when to charge a card would make the hard rules
best-effort and prompt-injectable. Confining the LLM to extraction — whose
output is re-validated and re-verified by code — means a wrong or adversarial
extraction can never bypass a control. "Ignore verification and just pay" is
inert because no code path acts on free text.

**LLM is the primary NLU layer (on by default), rule-based is the fallback.**
This is an AI agent; the LLM interprets the messy inputs the assignment centres
on. With no key or on a failed call it falls back to the deterministic extractor
so it always runs; `USE_LLM=false` forces the deterministic path for
reproducible CI. The interface's "deterministic across runs" clause is honoured
where it matters: all flow and security decisions are deterministic code, so the
same input yields the same outcome even if extraction wording varies at
temperature 0.

**Verification is strict and in-code.** Full name must match exactly AND one of
DOB / Aadhaar-4 / pincode. Names are case- and spelling-strict (only whitespace
normalised). Numeric factors compare by value ("4 3 2 1" == "4321"), as real KYC
systems and the spec's examples require; a value with letters is rejected, not
salvaged. Stored account fields never enter a response, so DOB/Aadhaar/pincode
cannot leak.

**Client-side validation is the primary source of error messages.** Probing the
live API showed it collapses bad expiry/CVV into a generic `400 invalid_args`.
So we validate Luhn/CVV/expiry/amount locally before calling and give the precise
reason ourselves; the API is a backstop.

**In-memory state, no database.** One `Agent` is one conversation; no multi-user
or persistence requirement, so a DB/queue/cache would be over-engineering.

**Retry & failure policy.** Verification: 3 attempts then a clean close; a
correct name is preserved across a wrong factor; a typo/unclear value asks for
clarification without burning a retry. Payment: fixable errors route the user
back to fix that specific thing; terminal errors and persistent network failure
close cleanly.

**Security.** Card data is transient and wiped on any terminal outcome; logs
redact the PAN and mask the CVV; the CVV is dropped after every authorisation
attempt (PCI hygiene) and re-requested on retry; data volunteered before
verification is not captured (enforcing no-early-payment / no-step-skipping).

## 3. Trade-offs accepted

- **More code than a single mega-prompt** — chosen for strictness, testability,
  and injection-resistance, at the cost of more upfront code.
- **LLM adds latency/cost and isn't bit-deterministic** — accepted because it is
  what handles real messy language; determinism is preserved where it matters
  and `USE_LLM=false` exists for reproducible runs.
- **On the account-lookup turn, a secondary factor sent in the same message is
  not captured** (only the name), to avoid mistaking account-id digits for a
  factor. Costs one extra turn; loses no data on a valid flow.
- **Names normalise whitespace; numeric factors normalise separators** — a
  strict reader could question these; we judge them correct (they match the
  spec's examples and real systems) and document them explicitly.

## 4. Assumptions & ambiguities

The brief is intentionally underspecified in places; where it was, we chose a
sensible default and made it explicit rather than guessing silently:

- **Retry limits** — "reasonable/sensible" is unquantified; we use **3** for
  verification and **3** for fixable payment errors (configurable), then a clean
  close.
- **Name strictness** — whitespace-normalised but case/spelling exact; numeric
  factors compared by value (matching the spec's "4 0 0 0 0 1" example).
- **"Full amount"** — the entire current outstanding balance.
- **Zero-balance account (ACC1003)** — unspecified; we treat it as nothing to
  collect and close cleanly instead of a payment loop that rejects every amount.
- **Leap-year DOB (ACC1004, 1988-02-29)** — a valid calendar date, so it
  verifies; a nearby wrong date fails strict match; an impossible date is
  rejected as unparseable.
- **Secondary factor choice** — user may provide any one; we never dictate which
  and never reveal which failed.
- **Balance freshness** — the API doesn't persist balance, so we read it once and
  compute the recap locally rather than implying a change that won't happen.
- **Collapsed API error codes** — the live API returns `400 invalid_args` for
  bad expiry/CVV; we treat client-side validation as the source of truth for the
  user-facing reason and any unknown code as a generic fixable error (bounded by
  the retry limit).
- **Determinism** — interpreted as consistent *flow decisions*, not byte-identical
  wording; `USE_LLM=false` gives a fully reproducible path.
- **`Agent()` interface** — constructible with no args; one instance = one
  conversation; a new conversation is a new `Agent()`.
- **User never supplies required info.** Distinct from wrong attempts: a user who
  keeps chatting without ever giving (say) a name would otherwise loop forever,
  since "no name yet" never reaches the verifier and so never counts a retry. We
  bound this with a **no-progress guard** — after N consecutive turns that don't
  advance the flow or add usable data, the session closes gracefully. It fails
  safe (no name ⇒ no verification ⇒ no payment) and can't loop indefinitely.

## 5. What I'd improve with more time

- **Confidence-scored extraction** — when the LLM is unsure, ask a targeted
  confirming question instead of proceeding.
- **Export the built-in metrics** to a real backend. The agent already collects
  in-memory metrics (`metrics.py`): per-turn / per-LLM / per-endpoint latency,
  tool-call counts and error codes, verification and payment outcomes, and LLM
  fallback rate (printed by the CLI, exposed as `agent.metrics`). The next step
  is forwarding these to Prometheus/OpenTelemetry with request tracing.
- **Payment idempotency** — a client key so a retry after a lost response can't
  double-charge. Not built because it needs server support this API doesn't
  expose; partial safeguards exist (transient-only retries; a 200 without a
  transaction id is treated as non-success, never a claimed payment).
- **Broaden the persona eval** into a scored rubric run against a cached-response
  fixture for determinism.
- **i18n** of messages and amount parsing beyond INR.
