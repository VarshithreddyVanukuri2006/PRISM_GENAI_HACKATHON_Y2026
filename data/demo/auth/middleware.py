from auth.tokens import decode_jwt, extract_token


class AuthMiddleware:
    """Check bearer tokens before protected handlers run."""

    def verify_token(self, request):
        """Validate a request token and return its claims."""
        token = extract_token(request.headers)
        return decode_jwt(token)


def protected_route(request):
    middleware = AuthMiddleware()
    return middleware.verify_token(request)
