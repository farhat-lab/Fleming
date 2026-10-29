"""AWS Cognito Hosted UI authentication helpers (OAuth2 Authorization Code flow)."""
import logging
import os
import time
import urllib.parse

import httpx
from jose import jwt
from jose.exceptions import JWTError

logger = logging.getLogger(__name__)

# --- Configuration (all via environment variables) ---
COGNITO_REGION = os.environ["COGNITO_REGION"]
COGNITO_USER_POOL_ID = os.environ["COGNITO_USER_POOL_ID"]
COGNITO_CLIENT_ID = os.environ["COGNITO_CLIENT_ID"]
COGNITO_CLIENT_SECRET = os.environ.get("COGNITO_CLIENT_SECRET")  # optional, omit for public clients
COGNITO_DOMAIN = os.environ["COGNITO_DOMAIN"]
# e.g. https://your-domain.auth.us-east-1.amazoncognito.com

APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://127.0.0.1:7860")
REDIRECT_PATH = "/auth/callback"
LOGOUT_REDIRECT_PATH = "/"

ISSUER = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}"
JWKS_URL = f"{ISSUER}/.well-known/jwks.json"
TOKEN_URL = f"{COGNITO_DOMAIN}/oauth2/token"
AUTHORIZE_URL = f"{COGNITO_DOMAIN}/login"
LOGOUT_URL = f"{COGNITO_DOMAIN}/logout"

_jwks_cache: dict = {"keys": None, "fetched_at": 0}
_JWKS_TTL_SECONDS = 3600


def _get_jwks() -> dict:
    """Fetch and cache the Cognito JSON Web Key Set used to verify token signatures."""
    now = time.time()
    if _jwks_cache["keys"] is None or now - _jwks_cache["fetched_at"] > _JWKS_TTL_SECONDS:
        response = httpx.get(JWKS_URL, timeout=10)
        response.raise_for_status()
        _jwks_cache["keys"] = response.json()
        _jwks_cache["fetched_at"] = now
    return _jwks_cache["keys"]


def build_login_url(state: str) -> str:
    """
    Build the Cognito Hosted UI login URL with the given state parameter. The state is used to
    prevent CSRF attacks and should be a securely generated random string.
    """
    redirect_uri = f"{APP_BASE_URL}{REDIRECT_PATH}"
    params = {
        "client_id": COGNITO_CLIENT_ID,
        "response_type": "code",
        "scope": "openid email profile",
        "redirect_uri": redirect_uri,
        "state": state,
    }
    url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"

    # No secrets in these params, so logging the exact URL makes Cognito "invalid_request"
    # errors easy to diagnose. If you see an "invalid_request" error, check that the
    # redirect_uri matches the the one configured in the Cognito App Client settings
    logger.info("Cognito Hosted UI login URL: %s", url)
    return url


def build_logout_url() -> str:
    params = {
        "client_id": COGNITO_CLIENT_ID,
        "logout_uri": f"{APP_BASE_URL}{LOGOUT_REDIRECT_PATH}",
    }
    return f"{LOGOUT_URL}?{urllib.parse.urlencode(params)}"


def exchange_code_for_tokens(code: str) -> dict:
    """Exchange an authorization code from the Hosted UI callback for tokens."""
    redirect_uri = f"{APP_BASE_URL}{REDIRECT_PATH}"
    data = {
        "grant_type": "authorization_code",
        "client_id": COGNITO_CLIENT_ID,
        "code": code,
        "redirect_uri": redirect_uri,
    }
    auth = None
    if COGNITO_CLIENT_SECRET:
        auth = (COGNITO_CLIENT_ID, COGNITO_CLIENT_SECRET)
    response = httpx.post(
        TOKEN_URL,
        data=data,
        auth=auth,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()



def verify_id_token(id_token: str, access_token: str) -> dict:
    """Validate signature, issuer, audience and expiry of a Cognito ID token; returns its claims."""
    jwks = _get_jwks()
    unverified_header = jwt.get_unverified_header(id_token)
    key = next((k for k in jwks["keys"] if k["kid"] == unverified_header["kid"]), None)
    if key is None:
        # Key rotation may have occurred; refresh cache once and retry.
        _jwks_cache["keys"] = None
        jwks = _get_jwks()
        key = next((k for k in jwks["keys"] if k["kid"] == unverified_header["kid"]), None)
        if key is None:
            raise JWTError("Unable to find matching JWKS key for token")

    claims = jwt.decode(
        id_token,
        key,
        algorithms=["RS256"],
        audience=COGNITO_CLIENT_ID,
        issuer=ISSUER,
        access_token=access_token
    )
    return claims
