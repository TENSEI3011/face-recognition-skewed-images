"""
auth_service.py
---------------
Lightweight JWT-based authentication service.
Credentials stored in web/backend/users.json (no DB required).
Default: admin / admin123
"""

import json
import hashlib
import secrets
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# ── Config ─────────────────────────────────────────────────────────────────────
SECRET_KEY   = "face-recog-uav-drdo-secret-2026"   # Change for production
ALGORITHM    = "HS256"
TOKEN_EXPIRE = 24  # hours

USERS_FILE = Path(__file__).parent.parent / "users.json"

# ── Lazy JWT import (python-jose) ──────────────────────────────────────────────
def _jwt():
    from jose import jwt
    return jwt

def _hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ── Default admin user bootstrap ───────────────────────────────────────────────
def _ensure_users_file():
    if not USERS_FILE.exists():
        USERS_FILE.write_text(json.dumps({
            "admin": {
                "password_hash": _hash_password("admin123"),
                "role": "admin",
                "created": datetime.utcnow().isoformat(),
            }
        }, indent=2))

def _load_users() -> dict:
    _ensure_users_file()
    return json.loads(USERS_FILE.read_text())

# ── Public API ──────────────────────────────────────────────────────────────────
def verify_credentials(username: str, password: str) -> bool:
    users = _load_users()
    user  = users.get(username)
    if not user:
        return False
    return user["password_hash"] == _hash_password(password)

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
    users = _load_users()
    if username not in users:
        return False
    users[username]["password_hash"] = _hash_password(new_password)
    USERS_FILE.write_text(json.dumps(users, indent=2))
    return True
