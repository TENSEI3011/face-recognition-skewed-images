"""
auth.py — Authentication router
POST /api/auth/login            → returns JWT token (raises 403 if password change required)
GET  /api/auth/me               → validate token, return username
POST /api/auth/logout           → client-side only (token invalidation hint)
POST /api/auth/change-password  → change password for logged-in user; clears must_change_password
"""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Optional

from web.backend.services.auth_service import (
    verify_credentials, create_token, verify_token,
    get_current_user, change_password, needs_password_change,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    new_password: str


@router.post("/login")
def login(req: LoginRequest):
    """Authenticate and return a JWT token.

    Returns HTTP 403 (not 200) when credentials are valid but the account
    still carries must_change_password=True.  The client should redirect to
    the change-password screen; the operator can then POST /change-password
    with a Bearer token obtained from /login's 403 response body, or simply
    call change-password after a successful subsequent login.

    Rationale: issuing a restricted token inside the 403 body lets the client
    hit the change-password endpoint without a second round-trip, while the
    HTTP 403 status makes it impossible to accidentally treat the response as
    a normal success and skip the rotation step.
    """
    if not verify_credentials(req.username, req.password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token = create_token(req.username)

    if needs_password_change(req.username):
        # Credentials are valid but the operator hasn't rotated the bootstrap
        # password yet.  Return a restricted token so the client can call
        # /change-password, but signal the requirement via HTTP 403.
        raise HTTPException(
            status_code=403,
            detail={
                "message": (
                    "You must change your password before accessing the system. "
                    "Use the token below to POST /api/auth/change-password."
                ),
                "must_change_password": True,
                "token": token,
            },
        )

    return {"token": token, "username": req.username, "message": "Login successful."}


@router.get("/me")
def me(authorization: str = Header(default="")):
    """Return current username if token is valid."""
    username = get_current_user(authorization)
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return {"username": username, "authenticated": True}


@router.post("/logout")
def logout():
    """Logout hint — actual invalidation is done client-side by clearing localStorage."""
    return {"message": "Logged out. Please clear your local token."}


@router.post("/change-password")
def change_pw(req: ChangePasswordRequest, authorization: str = Header(default="")):
    """Change password for the currently authenticated user.

    Also clears the must_change_password flag so subsequent logins succeed
    normally.  Accepts the restricted token issued by the login 403 response,
    making it usable as the first step after a first-run admin bootstrap.
    """
    username = get_current_user(authorization)
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    if len(req.new_password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters.")
    change_password(username, req.new_password)
    return {"message": "Password changed successfully."}
