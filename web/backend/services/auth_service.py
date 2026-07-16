"""
auth_service.py
---------------
Lightweight JWT-based authentication service.
Credentials stored in web/backend/users.json (no DB required).

Security notes
~~~~~~~~~~~~~~
* SECRET_KEY is loaded from the JWT_SECRET_KEY env var.  If unset a fresh
  random value is generated at startup — tokens survive a restart only when
  the var is pinned in .env / the deployment environment.
* Passwords are hashed with bcrypt (via passlib).  Accounts that still carry
  the old plain-SHA-256 hash are transparently migrated on first successful
  login so no manual reset is required.
* The bootstrap admin account is assigned a random password printed once to
  stdout; it carries must_change_password=True so the operator must rotate
  it before the JWT grants access to protected routes.
"""

import json
import hashlib
import logging
import os
import secrets
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── JWT secret — never fall back to a fixed default string ────────────────────
# Loading from env keeps the secret out of source control and makes it
# consistent across restarts.  A random fallback is safe for development
# but useless in production (all sessions die with the process).
_env_secret = os.getenv("JWT_SECRET_KEY", "")
if _env_secret:
    SECRET_KEY = _env_secret
else:
    SECRET_KEY = secrets.token_hex(32)
    logger.warning(
        "JWT_SECRET_KEY env var is not set — generated a random secret for "
        "this process.  All existing tokens will be invalidated on restart.  "
        "Set JWT_SECRET_KEY in your .env file for persistent sessions."
    )

ALGORITHM    = "HS256"
TOKEN_EXPIRE = 24  # hours

USERS_FILE = Path(__file__).parent.parent / "users.json"

# ── Lazy JWT import (python-jose) ──────────────────────────────────────────────
def _jwt():
    from jose import jwt
    return jwt

# ── Password hashing (bcrypt via passlib) ─────────────────────────────────────
# We use passlib's CryptContext so the scheme can be upgraded without touching
# every call-site.  The "deprecated='auto'" flag means passlib will
# automatically re-hash any password that is verified against an older scheme.
def _get_crypt_context():
    """Return the shared passlib CryptContext (imported lazily to keep startup fast)."""
    from passlib.context import CryptContext
    return CryptContext(schemes=["bcrypt"], deprecated="auto")

def _hash_password(pw: str) -> str:
    """Return a bcrypt hash of *pw* suitable for storage."""
    return _get_crypt_context().hash(pw)

def _is_bcrypt_hash(h: str) -> bool:
    """Return True if *h* looks like a bcrypt hash (starts with $2b$ or $2a$).

    Used to detect legacy SHA-256 hashes so we can migrate them on first login
    without forcing a password reset for every existing account.
    """
    return h.startswith(("$2b$", "$2a$", "$2y$"))

def _sha256_hash(pw: str) -> str:
    """Legacy SHA-256 hash — used only during the one-time migration check."""
    return hashlib.sha256(pw.encode()).hexdigest()

# ── Default admin user bootstrap ───────────────────────────────────────────────
def _ensure_users_file():
    """Create users.json with a random admin password if the file doesn't exist.

    The generated password is printed to stdout exactly once so the operator
    can retrieve it from the deployment logs.  The account is flagged with
    must_change_password=True; the login endpoint will reject token issuance
    until the operator has rotated the password via /api/auth/change-password.
    """
    if not USERS_FILE.exists():
        random_pw = secrets.token_urlsafe(12)
        print(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║              FIRST-RUN: ADMIN CREDENTIALS                   ║\n"
            "║                                                              ║\n"
            f"║  username : admin                                            ║\n"
            f"║  password : {random_pw:<50}║\n"
            "║                                                              ║\n"
            "║  You MUST change this password before using the system.      ║\n"
            "║  This message will NOT be shown again.                       ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n",
            flush=True,
        )
        USERS_FILE.write_text(json.dumps({
            "admin": {
                "password_hash": _hash_password(random_pw),
                "role": "admin",
                "must_change_password": True,
                "created": datetime.utcnow().isoformat(),
            }
        }, indent=2))

def _load_users() -> dict:
    _ensure_users_file()
    return json.loads(USERS_FILE.read_text())

def _save_users(users: dict) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2))

# ── Public API ──────────────────────────────────────────────────────────────────
def verify_credentials(username: str, password: str) -> bool:
    """Return True if *password* is correct for *username*.

    On success, transparently migrates legacy SHA-256 hashes to bcrypt so
    existing accounts are upgraded without any manual intervention.
    """
    users = _load_users()
    user  = users.get(username)
    if not user:
        return False

    stored_hash = user["password_hash"]

    if _is_bcrypt_hash(stored_hash):
        # Modern path: let passlib verify (and auto-rehash if the cost factor
        # has been bumped in the CryptContext).
        ctx = _get_crypt_context()
        ok, new_hash = ctx.verify_and_update(password, stored_hash)
        if not ok:
            return False
        if new_hash:
            # passlib decided to rehash (e.g. cost factor upgraded) — save it.
            users[username]["password_hash"] = new_hash
            _save_users(users)
        return True
    else:
        # Legacy path: verify against the old SHA-256 scheme once, then
        # immediately re-hash with bcrypt so the next login uses the modern path.
        if _sha256_hash(password) != stored_hash:
            return False
        users[username]["password_hash"] = _hash_password(password)
        _save_users(users)
        logger.info("Migrated legacy SHA-256 hash to bcrypt for user '%s'.", username)
        return True

def needs_password_change(username: str) -> bool:
    """Return True if *username* must change their password before accessing the API."""
    users = _load_users()
    user  = users.get(username, {})
    return bool(user.get("must_change_password", False))

def create_token(username: str) -> str:
    jwt = _jwt()
    expire = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> Optional[str]:
    """Return username if token is valid, else None."""
    try:
        jwt = _jwt()
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except Exception:
        return None

def get_current_user(authorization: str = "") -> Optional[str]:
    """Extract and verify token from 'Bearer <token>' header value."""
    if not authorization.startswith("Bearer "):
        return None
    return verify_token(authorization[7:])

def change_password(username: str, new_password: str) -> bool:
    """Hash *new_password* with bcrypt and persist it; clears must_change_password."""
    users = _load_users()
    if username not in users:
        return False
    users[username]["password_hash"] = _hash_password(new_password)
    # Clear the force-change flag once the user has chosen their own password.
    users[username].pop("must_change_password", None)
    _save_users(users)
    return True
