"""
auth.py — Authentication router
POST /api/auth/login       → returns JWT token
GET  /api/auth/me          → validate token, return username
POST /api/auth/logout      → client-side only (token invalidation hint)
POST /api/auth/change-password → change password for logged-in user
"""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Optional

from web.backend.services.auth_service import (
    verify_credentials, create_token, verify_token, get_current_user, change_password
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    new_password: str


@router.post("/login")
def login(req: LoginRequest):
    """Authenticate and return a JWT token."""
    if not verify_credentials(req.username, req.password):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = create_token(req.username)
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
    """Change password for the currently authenticated user."""
    username = get_current_user(authorization)
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    if len(req.new_password) < 4:
        raise HTTPException(status_code=400, detail="Password must be at least 4 characters.")
    change_password(username, req.new_password)
    return {"message": "Password changed successfully."}
