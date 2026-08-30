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

This is an LLM agent: the **LLM extractor is the primary NLU layer and is on by
default**. Provide an OpenAI key (e.g. in a local `.env` file) and it interprets
the messy natural language; deterministic code owns verification, validation,
and the flow.

```bash
# .env  (git-ignored)
OPENAI_API_KEY=sk-...
USE_LLM=true
```

If no key is present, or an LLM call fails, the agent transparently falls back
to a deterministic rule-based extractor so it always runs and never breaks a
turn. To force the offline path (e.g. for fully reproducible CI), set
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

The offline harness reports **14/14 scenarios, 73/73 checks, and tool-call
correctness 16/16**. Payment-failure coverage includes invalid card, expired
card, server-side insufficient_balance, and server-side invalid_amount. The LLM
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
  tools/
    payment_api.py           # HTTP client, retry + error normalisation
tests/                       # pytest suite + mock API
eval/                        # scenario harness, metrics, LLM personas
```

---

## Sample conversations

Captured from a live run of the agent using the **LLM extractor** and the
**real payment API** (hence the real transaction IDs). This is the default
mode - the LLM interprets the messy natural language; deterministic code owns
verification, validation, and the flow.

### 1. Successful end-to-end payment
```
agent> Hello! I can help you clear your outstanding balance. To get started, could you share your account ID?
you  > yeah my account number is ACC1001 I think
agent> Thanks. Could you please confirm your full name as it appears on the account?
you  > it's Nithin, Nithin Jain
agent> I have your name. Now I just need ONE of these to verify you: date of birth, last 4 digits of Aadhaar, or pincode.
you  > I was born on 14th May 1990
agent> Identity verified, thank you! Your current outstanding balance is ₹1,250.75. How much would you like to pay today? You can pay the full amount or a partial amount.
you  > I want to pay a thousand rupees
agent> Got it - ₹1,000.00. Now I'll need your card details: the card number, expiry (month and year), CVV, and the name on the card.
you  > the card number is 4532 0151 1283 0366, expires December 2027, CVV is one two three, name on card Nithin Jain
agent> All done! Here's a quick recap:
       - Account: ACC1001
       - Amount paid: ₹1,000.00
       - Transaction ID: txn_1788089911446_xdsjaun
       Your remaining balance is ₹250.75. Thank you for your payment. This session is now complete - take care!
```

### 2. Verification failure (retries exhausted)

Note how the correct name is kept across attempts - only the mismatched factor
is re-requested - and a failed *factor* does not force re-typing the name.
```
agent> Hello! I can help you clear your outstanding balance. To get started, could you share your account ID?
you  > ACC1001
agent> Thanks. Could you please confirm your full name as it appears on the account?
you  > Nithin Jain
agent> I have your name. Now I just need ONE of these to verify you: date of birth, last 4 digits of Aadhaar, or pincode.
you  > my dob is 2000-01-01
agent> Thanks - I've got your name, but that detail didn't match our records. You have 2 attempts remaining. Could you share one of: date of birth, Aadhaar last 4, or pincode?
you  > aadhaar 1111
agent> Thanks - I've got your name, but that detail didn't match our records. You have 1 attempt remaining. Could you share one of: date of birth, Aadhaar last 4, or pincode?
you  > pincode 999999
agent> I'm sorry, but I couldn't verify your identity after several attempts, so I can't proceed for security reasons. Please contact support for help. Take care!
```

### 3. Payment failure (expired card caught locally, then recovery)

The expired card is caught by client-side validation *before* any API call, the
user gets a clear reason, and the flow recovers with a valid card. (An invalid
card number - failing the Luhn check - is handled the same way.)
```
agent> Hello! I can help you clear your outstanding balance. To get started, could you share your account ID?
you  > acc1001
agent> Thanks. Could you please confirm your full name as it appears on the account?
you  > Nithin Jain
agent> I have your name. Now I just need ONE of these to verify you: date of birth, last 4 digits of Aadhaar, or pincode.
you  > dob 1990-05-14
agent> Identity verified, thank you! Your current outstanding balance is ₹1,250.75. How much would you like to pay today? You can pay the full amount or a partial amount.
you  > pay 500
agent> Got it - ₹500.00. Now I'll need your card details: the card number, expiry (month and year), CVV, and the name on the card.
you  > 4532 0151 1283 0366, expiry 01/2020, cvv 123, name Nithin Jain
agent> That card appears to have expired. Please use a card that is still valid.
you  > ok try 4532 0151 1283 0366, 12/27, cvv 123, name Nithin Jain
agent> All done! Here's a quick recap:
       - Account: ACC1001
       - Amount paid: ₹500.00
       - Transaction ID: txn_1788089933857_yd8d9q1
       Your remaining balance is ₹750.75. Thank you for your payment. This session is now complete - take care!
```

### 4. Edge case: leap-year DOB + out-of-order info
```
agent> Hello! I can help you clear your outstanding balance. To get started, could you share your account ID?
you  > my account is ACC1004 and my name is Rahul Mehta
agent> Thanks, Rahul Mehta. To verify your identity, could you share ONE of the following: your date of birth, the last 4 digits of your Aadhaar, or your pincode?
you  > 29th February 1988
agent> Identity verified, thank you! Your current outstanding balance is ₹3,200.50. How much would you like to pay today? You can pay the full amount or a partial amount.
you  > just clear the full amount
agent> Got it - ₹3,200.50. Now I'll need your card details: the card number, expiry (month and year), CVV, and the name on the card.
you  > 4532 0151 1283 0366, 12/27, cvv 123, cardholder Rahul Mehta
agent> All done! Here's a quick recap:
       - Account: ACC1004
       - Amount paid: ₹3,200.50
       - Transaction ID: txn_1788089941476_25sxedl
       Your balance is now fully cleared. Thank you for your payment. This session is now complete - take care!
```

In example 4 the leap-year date `1988-02-29` is accepted as valid and matches
the account, the name is captured out-of-order alongside the account id (so the
agent greets by name and only asks for the secondary factor), and "clear the
full amount" is resolved to the exact outstanding balance.
