"""
WHAT THIS FILE IS (in one line):
    Every sentence the agent ever says to the user lives here.

WHY ALL IN ONE PLACE:
    - One spot to read/tweak the wording and tone.
    - One spot to guarantee the hard rule: we NEVER print the stored DOB /
      Aadhaar / pincode. (Notice none of these functions even take those values -
      the only account detail they accept is the balance and the account id.)

HOW IT'S USED:
    orchestrator.py picks the right message from here each turn. Some are plain
    strings (constants); some are functions that fill in a number (like the
    balance or the amount). The optional phraser.py may reword the *safe* ones
    to sound friendlier, but sensitive ones (balance/recap/errors) are shown
    exactly as written here.
"""

from __future__ import annotations

from .validators import format_currency   # turns 1250.75 into "₹1,250.75"


class Responses:
    # Below: constants (fixed text) and small functions (text with a number
    # filled in). Grouped by the step they belong to.
    # --- greeting / account ------------------------------------------------ #
    GREETING = "Hello! I can help you clear your outstanding balance. To get started, could you share your account ID?"

    ASK_ACCOUNT_AGAIN = "I didn't catch a valid account ID. It usually looks like 'ACC1001'. Could you share it?"

    @staticmethod
    def account_not_found(remaining: int) -> str:
        if remaining > 0:
            return (
                "I couldn't find an account with that ID. Please double-check it "
                "and share it again."
            )
        return (
            "I still couldn't find that account, so I'm unable to continue right "
            "now. Please contact support if you need further help. Take care!"
        )

    # --- identity / verification ------------------------------------------ #
    ASK_NAME = "Thanks. Could you please confirm your full name as it appears on the account?"

    ASK_SECONDARY = (
        "Thanks, {name}. To verify your identity, could you share ONE of the "
        "following: your date of birth, the last 4 digits of your Aadhaar, or "
        "your pincode?"
    )

    ASK_NAME_FIRST = (
        "Before I can verify you, I'll need your full name. Could you share it?"
    )

    ASK_SECONDARY_ONLY = (
        "I have your name. Now I just need ONE of these to verify you: date of "
        "birth, last 4 digits of Aadhaar, or pincode."
    )

    UNCLEAR_SECONDARY = (
        "Hmm, that number doesn't look like a date of birth, a 4-digit Aadhaar, "
        "or a 6-digit pincode. Could you clarify - for example 'Aadhaar 4321', "
        "'pincode 400001', or 'DOB 1990-05-14'?"
    )

    @staticmethod
    def verification_failed(remaining: int) -> str:
        return (
            "Sorry, I couldn't verify your identity with those details. "
            f"You have {remaining} attempt{'s' if remaining != 1 else ''} "
            "remaining. Please re-share your full name and one of: date of birth, "
            "Aadhaar last 4, or pincode."
        )

    @staticmethod
    def verification_failed_factor(remaining: int) -> str:
        # Name already matched; we keep it and ask only for a correct factor.
        # Deliberately does not reveal which factor or the stored value.
        return (
            "Thanks - I've got your name, but that detail didn't match our "
            f"records. You have {remaining} attempt{'s' if remaining != 1 else ''} "
            "remaining. Could you share one of: date of birth, Aadhaar last 4, "
            "or pincode?"
        )

    VERIFICATION_LOCKED = (
        "I'm sorry, but I couldn't verify your identity after several attempts, "
        "so I can't proceed for security reasons. Please contact support for help. "
        "Take care!"
    )

    # --- balance / amount -------------------------------------------------- #
    @staticmethod
    def verified_balance(balance: float) -> str:
        return (
            "Identity verified, thank you! Your current outstanding balance is "
            f"{format_currency(balance)}. How much would you like to pay today? "
            "You can pay the full amount or a partial amount."
        )

    ASK_AMOUNT = "How much would you like to pay? You can pay the full amount or a partial amount."

    @staticmethod
    def nothing_to_pay(account_id: str) -> str:
        return (
            "Identity verified, thank you! Good news - account "
            f"{account_id} has no outstanding balance, so there's nothing to "
            "pay right now. Have a great day - this session is now complete!"
        )

    @staticmethod
    def amount_confirmed(amount: float) -> str:
        # Confirm the amount, then ask for the FIRST card field only. We collect
        # card details one at a time for a natural, call-centre-like flow.
        return (
            f"Got it - {format_currency(amount)}. Let's take your card details. "
            "First, what's your card number?"
        )

    # --- card (collected one field at a time) ------------------------------ #
    # A friendly prompt per field, asked in this order.
    _CARD_FIELD_PROMPTS = {
        "card_number": "What's your card number?",
        "expiry": "Thanks. What's the card's expiry (month and year)?",
        "cvv": "Got it. And the CVV (the 3 or 4 digit code)?",
        "cardholder_name": "Almost there - what's the name as it appears on the card?",
    }

    @staticmethod
    def ask_card_field(field: str) -> str:
        """Ask for a single card field (one-at-a-time collection)."""
        return Responses._CARD_FIELD_PROMPTS.get(
            field, "Could you share the remaining card detail?"
        )

    # --- confirmation before charging -------------------------------------- #
    @staticmethod
    def confirm_payment(amount: float, card_last4: str, exp_month: int, exp_year: int) -> str:
        # Safe summary only: amount + card LAST 4 + expiry. Never the full card
        # number or CVV. Shown before we charge so the user can back out.
        return (
            "Please confirm before I process the payment:\n"
            f"- Amount: {format_currency(amount)}\n"
            f"- Card ending {card_last4}, expiry {exp_month:02d}/{exp_year}\n"
            "Shall I go ahead? (yes / no)"
        )

    CONFIRM_UNCLEAR = (
        "Just to be safe, I didn't catch a clear yes or no. Shall I go ahead "
        "with the payment? Please reply 'yes' to proceed or 'no' to cancel."
    )

    PAYMENT_CANCELLED = (
        "No problem - I've cancelled this payment and won't charge your card. "
        "This session is now complete. Take care!"
    )

    # --- payment outcome --------------------------------------------------- #
    @staticmethod
    def payment_success(
        amount: float,
        transaction_id: str,
        account_id: str,
        remaining_balance: float,
    ) -> str:
        # Full recap: what was paid, on which account, the transaction id, and
        # the remaining balance for this session, then a clean close.
        if round(remaining_balance, 2) <= 0:
            balance_line = "Your balance is now fully cleared."
        else:
            balance_line = (
                f"Your remaining balance is {format_currency(remaining_balance)}."
            )
        return (
            f"All done! Here's a quick recap:\n"
            f"- Account: {account_id}\n"
            f"- Amount paid: {format_currency(amount)}\n"
            f"- Transaction ID: {transaction_id}\n"
            f"{balance_line} "
            "Thank you for your payment. This session is now complete - take care!"
        )

    @staticmethod
    def payment_retryable(reason: str, remaining: int) -> str:
        return (
            f"{reason} You have {remaining} attempt{'s' if remaining != 1 else ''} "
            "left."
        )

    @staticmethod
    def payment_terminal(reason: str) -> str:
        return (
            f"{reason} I'm unable to complete the payment right now, so I'll close "
            "this session. Please try again later or contact support. Take care!"
        )

    # --- generic ----------------------------------------------------------- #
    NETWORK_ERROR = (
        "I'm having trouble reaching our payment system right now. Please try "
        "again in a moment."
    )

    NETWORK_ERROR_TERMINAL = (
        "I'm still unable to reach our payment system, so I'll close this session "
        "for now. Please try again later. Take care!"
    )

    FALLBACK = (
        "Sorry, I didn't quite catch that. Could you rephrase or provide the "
        "requested information?"
    )

    # General "I didn't understand this turn" clarifications, one per step, so
    # the agent always guides the user rather than silently re-prompting.
    DIDNT_UNDERSTAND_ACCOUNT = (
        "Sorry, I didn't catch an account ID in that. It usually looks like "
        "'ACC1001' - could you share it?"
    )
    DIDNT_UNDERSTAND_IDENTITY = (
        "Sorry, I didn't catch that. To verify you I need your full name and one "
        "of: date of birth, last 4 digits of Aadhaar, or pincode."
    )
    DIDNT_UNDERSTAND_AMOUNT = (
        "Sorry, I didn't catch an amount there. How much would you like to pay? "
        "For example '500' or 'the full amount'."
    )
    DIDNT_UNDERSTAND_CARD = (
        "Sorry, I didn't catch any card details there. Please share the card "
        "number, expiry (month and year), CVV, and the name on the card."
    )

    USER_QUIT = "No problem - I've cancelled this session. Feel free to come back anytime. Take care!"

    NO_PROGRESS = (
        "It looks like we're not able to get the details needed to continue, so "
        "I'll close this session for now. Please start again when you have the "
        "required information handy. Take care!"
    )

    ALREADY_CLOSED = "This session has ended. Please start a new conversation if you'd like to make a payment."


# When a payment fails, this table tells the orchestrator two things per error:
#   1) the message to show the user, and
#   2) is it TERMINAL? (True = give up/close, False = user can fix it and retry)
# Example: "invalid_card" -> tell them to try another card, and it's fixable
# (False), so we let them re-enter the card. "account_not_found" is terminal.
PAYMENT_ERROR_GUIDANCE: dict[str, tuple[str, bool]] = {
    "invalid_card": ("That card was declined as invalid. Please check the number and try a different card.", False),
    "invalid_cvv": ("The CVV wasn't accepted. Please re-check the CVV.", False),
    "invalid_expiry": ("The card expiry wasn't accepted. Please re-check the expiry date.", False),
    "invalid_args": ("Some card details weren't accepted. Please re-check the card number, expiry, and CVV.", False),
    "invalid_amount": ("That payment amount wasn't valid. Please provide a valid amount.", False),
    "insufficient_balance": ("That amount exceeds your outstanding balance. Please enter a smaller amount.", False),
    "account_not_found": ("I could no longer find that account.", True),
}
