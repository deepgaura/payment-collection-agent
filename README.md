# Payment Collection AI Agent

A conversational agent that runs an end-to-end payment-collection flow over
chat: greet → look up account → verify identity → share balance → collect a
payment → process it → recap and close.

The design principle throughout is **LLM for understanding, deterministic code
for deciding**. The LLM is used only as a natural-language extraction layer; all
control flow, verification, validation, retry accounting, and API calls are
deterministic Python. This keeps security-critical logic strict, reproducible,
and impossible to bypass through prompt injection. See `DESIGN.md` for the full
rationale.

---

## Quick start

```bash
# 1. Install runtime dependencies
pip install -r requirements.txt

# 2. Run the interactive CLI
python cli.py
```

This is an LLM agent: the **LLM is the primary NLU layer (on by default)** and
also powers an optional conversational tone layer. It supports two providers -
**Anthropic Claude on Google Vertex (default)** or **OpenAI** - selected by
`LLM_PROVIDER`. Configure via a local `.env` (git-ignored); see `.env.example`.

```bash
# .env  — Claude on Vertex (uses `gcloud auth application-default login`)
LLM_PROVIDER=vertex
ANTHROPIC_VERTEX_PROJECT_ID=your-gcp-project
CLOUD_ML_REGION=europe-west1
ANTHROPIC_MODEL=claude-opus-4-5@20251101

# …or OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-...
```

Deterministic code owns verification, validation, and the flow. If no provider
is configured, or an LLM call fails, the agent transparently falls back to a
deterministic rule-based extractor and template replies, so it always runs and
never breaks a turn. To force the fully offline/reproducible path, set
`USE_LLM=false`.

On determinism: all flow, verification, validation, and API decisions are
deterministic code - only free-text understanding uses the LLM (at temperature
0), so the same input drives the same flow decisions across runs.

### Programmatic use (the evaluated interface)

```python
from agent import Agent

agent = Agent()
print(agent.next("Hi"))
# {"message": "Hello! I can help you clear your outstanding balance. ..."}
print(agent.next("my account is ACC1001"))
```

`Agent.next(user_input: str) -> {"message": str}` is the exact interface the
evaluator uses. `Agent()` takes no required arguments, each instance is one
conversation and holds all state internally between calls, and no external setup
is needed between turns.

---

## Configuration

All tunables are environment variables (see `src/payment_agent/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `USE_LLM` | `true` | LLM extractor is the default; set `false` to force the deterministic rule-based path |
| `OPENAI_API_KEY` | – | Enables the LLM extractor |
| `LLM_MODEL` | `gpt-4o-mini` | Extraction model |
| `PAYMENT_API_BASE_URL` | (assignment URL) | API base |
| `PAYMENT_API_TIMEOUT` | `20` | Per-request timeout (s) |
| `MAX_VERIFICATION_ATTEMPTS` | `3` | Verification retry limit |
| `MAX_PAYMENT_ATTEMPTS` | `3` | Payment fix-up retry limit |

---

## Tests and evaluation

```bash
# Unit + flow tests (offline, deterministic) - 86 tests
pip install -r requirements-dev.txt
python -m pytest

# Scenario evaluation harness with metrics (offline, deterministic)
python -m eval.run_eval --verbose

# Same scenarios against the real API
python -m eval.run_eval --live

# LLM persona simulator: an LLM role-plays users end-to-end (needs a key)
python -m eval.personas
```

The offline harness reports **15/15 scenarios, 85/85 checks, and tool-call
correctness 20/20**. Payment-failure coverage includes invalid card, expired
card, server-side insufficient_balance, server-side invalid_amount, and a user
declining at the confirmation step. The LLM
persona simulator reports **3/3 personas correct** (terse legit and chatty legit
both pay; impostor is rejected and never charged). See `EVALUATION.md` for the
full approach and what "correct" means per step, and `DESIGN.md` for the
architecture and design decisions.

---

## Repository layout

```
agent.py                     # root facade exposing the required Agent class
cli.py                       # interactive REPL
src/payment_agent/
  agent.py                   # Agent: wires extractor + orchestrator + state
  orchestrator.py            # deterministic state machine (all decisions)
  state.py                   # SessionState, Step enum, Account/Card/Claim
  verifier.py                # strict name + secondary-factor verification
  validators.py              # Luhn, CVV, expiry, amount, strict date parsing
  responses.py               # safe user-facing message templates
  config.py                  # environment-driven configuration
  extractors/
    base.py                  # Extractor interface + ExtractionResult
    llm.py                   # LLM NLU (primary, on by default; engineered prompt)
    rule_based.py            # deterministic NLU (fallback + reproducible CI path)
  llm_client.py              # provider-agnostic client: Claude/Vertex or OpenAI
  phraser.py                 # optional conversational tone layer (safe messages only)
  tools/
    payment_api.py           # HTTP client, retry + error normalisation
tests/                       # pytest suite + mock API
eval/                        # scenario harness, metrics, LLM personas
```

---

## Sample conversations

Captured from **live runs** using Claude (via Vertex) and the **real payment
API** (hence the real transaction IDs). The LLM interprets the messy natural
language and gives the replies their conversational tone; deterministic code
owns verification, validation, the flow, and every sensitive/transactional
message (balance, transaction id, errors are exact templates, never
LLM-generated). Card details are collected one field at a time.

### 1. Successful end-to-end payment (card collected one field at a time)
```
agent> Hello! I can help you clear your outstanding balance. To get started, could you share your account ID?
you  > hi, i'd like to clear my dues
agent> Happy to help you clear your dues! I didn't catch an account ID though — it usually looks like 'ACC1001,' could you share yours?
you  > yeah my account is ACC1001
agent> Great, thanks! Could you please confirm your full name as it appears on the account?
you  > it's Nithin Jain
agent> Got it, Nithin! Now I just need one of these to verify you: your date of birth, last 4 digits of Aadhaar, or your pincode.
you  > I was born on 14th May 1990
agent> Identity verified, thank you! Your current outstanding balance is ₹1,250.75. How much would you like to pay today? You can pay the full amount or a partial amount.
you  > let me pay a thousand
agent> Great, ₹1,000.00 it is! Let's get your card details - can you start with your card number?
you  > 4532 0151 1283 0366
agent> Got it, thanks! What's the expiry date on that card—month and year?
you  > expires December 2027
agent> Perfect, December 2027 noted. And what's the CVV — that 3 or 4 digit code on your card?
you  > cvv is one two three
agent> Thanks for that! Just need the name as it appears on the card.
you  > Nithin Jain
agent> Please confirm before I process the payment:
       - Amount: ₹1,000.00
       - Card ending 0366, expiry 12/2027
       Shall I go ahead? (yes / no)
you  > yes go ahead
agent> All done! Here's a quick recap:
       - Account: ACC1001
       - Amount paid: ₹1,000.00
       - Transaction ID: txn_1788441198907_exg39s6
       Your remaining balance is ₹250.75. Thank you for your payment. This session is now complete - take care!
```

### 2. Verification failure (retries exhausted)

The correct name is kept across attempts - only the mismatched factor is
re-requested - and the agent never reveals which stored value it checked.
```
agent> Hello! I can help you clear your outstanding balance. To get started, could you share your account ID?
you  > ACC1001
agent> Got it, thanks! Could you please confirm your full name exactly as it appears on the account?
you  > Nithin Jain
agent> Thanks, Nithin! I just need one thing to verify you—your date of birth, last 4 digits of your Aadhaar, or your pincode.
you  > my dob is 2000-01-01
agent> Thanks for that! I've got your name down, but unfortunately that date of birth didn't match what we have on file. You've got 2 attempts left — could you try your Aadhaar last 4 digits or your pincode instead?
you  > aadhaar 1111
agent> Thanks for that! Unfortunately, those digits didn't match what we have on file. You've got one attempt left — could you try your date of birth, Aadhaar last 4, or pincode instead?
you  > pincode 999999
agent> I'm sorry, but I couldn't verify your identity after several attempts, so I can't proceed for security reasons. Please contact support for help. Take care!
```

### 3. Payment failure (expired card caught locally, then recovery)

The expired card is caught by client-side validation *before* any API call, the
user gets a clear reason, and the flow recovers. (An invalid card number -
failing the Luhn check - is handled the same way.)
```
agent> Hello! I can help you clear your outstanding balance. To get started, could you share your account ID?
you  > acc1001
agent> Perfect, thank you! Could you please confirm your full name as it appears on the account?
you  > Nithin Jain
agent> Thanks, Nithin! Now I just need one of these to verify you: your date of birth, last 4 digits of Aadhaar, or your pincode.
you  > dob 1990-05-14
agent> Identity verified, thank you! Your current outstanding balance is ₹1,250.75. How much would you like to pay today? You can pay the full amount or a partial amount.
you  > pay 500
agent> Noted – ₹500.00. Let's get your card details. What's your card number?
you  > 4532 0151 1283 0366
agent> Got it, thanks! What's the expiry date on that card—month and year?
you  > expiry 01/2020
agent> Got it, thanks! And could I get the CVV—that's the 3 or 4 digit code on your card?
you  > 123
agent> Got it! And what's the name as it appears on the card?
you  > Nithin Jain
agent> Got it, Nithin! Unfortunately, it looks like that card has expired—could you please use one that's still valid?
you  > oh, the expiry is 12/27
agent> All done! Here's a quick recap:
       - Account: ACC1001
       - Amount paid: ₹500.00
       - Transaction ID: txn_1788441343677_5vs34ux
       Your remaining balance is ₹750.75. Thank you for your payment. This session is now complete - take care!
```

### 4. Edge case: zero-balance account (ACC1003)

A verified account with nothing owed is closed cleanly, instead of entering a
payment loop that would reject every amount.
```
agent> Hello! I can help you clear your outstanding balance. To get started, could you share your account ID?
you  > ACC1003
agent> Got it, thanks! Could you please confirm your full name as it appears on the account?
you  > Priya Agarwal
agent> Thanks, Priya! Now I just need one of the following to verify your account: your date of birth, last 4 digits of your Aadhaar, or your pincode.
you  > dob 1992-08-10
agent> Identity verified, thank you! Good news - account ACC1003 has no outstanding balance, so there's nothing to pay right now. Have a great day - this session is now complete!
```

In example 4 the leap-year date `1988-02-29` is accepted as valid and matches
the account, the name is captured out-of-order alongside the account id (so the
agent greets by name and only asks for the secondary factor), and "clear the
full amount" is resolved to the exact outstanding balance.
