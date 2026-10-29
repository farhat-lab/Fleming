"""Authentication middleware that protects every request behind AWS Cognito Hosted UI."""
import secrets
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.types import ASGIApp

from jose.exceptions import JWTError

from . import cognito_auth

# Paths that must remain reachable without a session so the login flow itself can complete.
PUBLIC_PATHS = {
    cognito_auth.REDIRECT_PATH,
    "/auth/login",
    "/auth/logout",
    "/healthz",
}


class CognitoAuthMiddleware(BaseHTTPMiddleware):
    """
    This middleware blocks every request unless the user has a valid session.
    Unauthenticated requests are redirected to the Cognito Hosted UI login page. After successful
    login, Cognito redirects back to the app with a valid session containing the user's ID token
    and access token. The middleware verifies the ID token on every request to ensure it is valid
    and not expired. If the token is invalid or expired, the session is cleared and the user is 
    redirected to the login page again.

    Use mount_gradio_app() to mount the Gradio app after this middleware is added, so that the 
    Gradio UI and its API are also protected by CognitoAuthMiddleware.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        id_token = request.session.get("id_token")
        access_token = request.session.get("access_token")
        claims = None
        if id_token:
            try:
                claims = cognito_auth.verify_id_token(id_token, access_token)
            except JWTError:
                claims = None

        if claims is None:
            request.session.clear()
            state = secrets.token_urlsafe(16)
            request.session["oauth_state"] = state
            request.session["post_login_redirect"] = str(request.url.path)
            return RedirectResponse(url=cognito_auth.build_login_url(state))

        # Reject expired tokens even if signature/issuer/audience checks passed.
        if claims.get("exp", 0) < time.time():
            request.session.clear()
            state = secrets.token_urlsafe(16)
            request.session["oauth_state"] = state
            return RedirectResponse(url=cognito_auth.build_login_url(state))

        request.state.user = claims
        return await call_next(request)
