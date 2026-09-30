import jwt


def extract_token(headers):
    """Read a bearer token from request headers."""
    return headers.get("Authorization", "").removeprefix("Bearer ")


def decode_jwt(token):
    """Decode and validate a signed JWT."""
    return jwt.decode(token, "development-secret", algorithms=["HS256"])
