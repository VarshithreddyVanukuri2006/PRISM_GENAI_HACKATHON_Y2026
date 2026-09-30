def validate_payment(amount, currency):
    """Reject invalid payment requests before charging."""
    if amount <= 0 or not currency:
        raise ValueError("invalid payment")


def charge_card(card, amount):
    """Submit a card charge to the payment provider."""
    return {"card": card, "amount": amount, "status": "charged"}


def process_payment(card, amount, currency):
    validate_payment(amount, currency)
    return charge_card(card, amount)
