import hashlib


def hash_password(password, salt):
    """Hash a user's password with a salt before storing credentials."""
    salted_password = (salt + password).encode("utf-8")
    return hashlib.sha256(salted_password).hexdigest()
