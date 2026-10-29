"""
FastAPI entrypoint that protects the Gradio app with AWS Cognito Hosted UI authentication.

Every route -- including the Gradio UI, its static assets and its predict API -- is protected
by CognitoAuthMiddleware. Unauthenticated requests are redirected to the Cognito Hosted UI;
only /auth/login, /auth/callback, /auth/logout and /healthz remain public.
"""
import logging
import os
import secrets

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from . import cognito_auth
from .auth_middleware import CognitoAuthMiddleware
from .run_fleming_interface import demo

# Without this, uvicorn's logging config suppresses our INFO-level auth diagnostics.
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(CognitoAuthMiddleware)

# Signs the session cookie; sessions hold only the id_token, never long-lived credentials.
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ["SESSION_SECRET_KEY"],
    same_site="lax",
    https_only=os.environ.get("SESSION_COOKIE_SECURE", "true").lower() == "true",
)


@app.get("/healthz")
async def healthz():
    """
    Health check endpoint that returns a simple JSON response indicating the service is running.
    This endpoint is publicly accessible and does not require authentication.
    """
    return {"status": "ok"}


@app.get("/auth/login")
async def login(request: Request):
    """
    Redirects the user to the Cognito Hosted UI login page. If the user is already logged in,
    Cognito will immediately redirect back to the app with a valid session.
    """
    state = secrets.token_urlsafe(16)
    request.session["oauth_state"] = state
    return RedirectResponse(url=cognito_auth.build_login_url(state), status_code=302)



@app.get(cognito_auth.REDIRECT_PATH)
async def auth_callback(request: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    """
    Handles the redirect from Cognito after the user logs in. Exchanges the authorization code for 
    tokens and stores them in the session. If the user is already logged in, this route will be 
    called immediately after /auth/login with a valid session.
    """
    if error:
        return RedirectResponse(url="/auth/login")

    expected_state = request.session.pop("oauth_state", None)
    if not code or not state or state != expected_state:
        return RedirectResponse(url="/auth/login")

    tokens = cognito_auth.exchange_code_for_tokens(code)
    claims = cognito_auth.verify_id_token(tokens["id_token"], tokens["access_token"])

    request.session.clear()
    request.session["id_token"] = tokens["id_token"]
    request.session["access_token"] = tokens["access_token"]
    request.session["user_email"] = claims.get("email")

    redirect_to = request.session.pop("post_login_redirect", "/")
    return RedirectResponse(url=redirect_to)


@app.get("/auth/logout")
async def logout(request: Request):
    """
    Logs the user out of the app and redirects to the Cognito Hosted UI logout page, which will
    clear the user's Cognito session and redirect back to the app's home page.
    """
    request.session.clear()
    return RedirectResponse(url=cognito_auth.build_logout_url())


# Mounting this app after the middleware and routes above ensures the Gradio UI and its API also
# pass through CognitoAuthMiddleware, since FastAPI middleware wraps the whole ASGI app.
# https://gradio.app/docs/gradio/mount_gradio_app
app = gr.mount_gradio_app(app, demo, path="/")
