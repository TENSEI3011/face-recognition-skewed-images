"""
watchlist.py — Watchlist + Alert router

PRIMARY:  MongoDB Atlas  (watchlist collection)
FALLBACK: web/backend/watchlist.json (if MongoDB not connected)

Endpoints:
  GET    /api/watchlist         → list all watchlisted names
  POST   /api/watchlist         → add a name
  DELETE /api/watchlist/{name}  → remove a name
  GET    /api/watchlist/hits    → recent watchlist hit events from audit log
"""

import json
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/watchlist", tags=["watchlist"])

_WATCHLIST_FILE = Path(__file__).resolve().parents[1] / "watchlist.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_local() -> list[str]:
    if not _WATCHLIST_FILE.exists():
        _WATCHLIST_FILE.write_text("[]")
    return json.loads(_WATCHLIST_FILE.read_text())


def _save_local(names: list[str]) -> None:
    _WATCHLIST_FILE.write_text(json.dumps(sorted(set(names)), indent=2))


def _get_db():
    try:
        from web.backend.services.mongo_service import get_db
        return get_db()
    except Exception:
        return None


def _load() -> list[str]:
    db = _get_db()
    if db is not None:
        try:
            docs = list(db["watchlist"].find({}, {"name": 1, "_id": 0}))
            return [d["name"] for d in docs]
        except Exception:
            pass
    return _load_local()


def _add(name: str) -> None:
    db = _get_db()
    if db is not None:
        try:
            db["watchlist"].update_one(
                {"name": name},
                {"$setOnInsert": {"name": name, "added_at": datetime.utcnow().isoformat()}},
                upsert=True,
            )
            return
        except Exception:
            pass
    # Fallback
    names = _load_local()
    if name not in names:
        names.append(name)
        _save_local(names)


def _remove(name: str) -> bool:
    db = _get_db()
    if db is not None:
        try:
            result = db["watchlist"].delete_one({"name": {"$regex": f"^{name}$", "$options": "i"}})
            return result.deleted_count > 0
        except Exception:
            pass
    # Fallback
    names = _load_local()
    new_names = [n for n in names if n.lower() != name.lower()]
    if len(new_names) == len(names):
        return False
    _save_local(new_names)
    return True


def is_watchlisted(name: str) -> bool:
    """Check if a name is on the watchlist (case-insensitive)."""
    return name.lower() in [n.lower() for n in _load()]


# ── Models ────────────────────────────────────────────────────────────────────

class WatchlistAddRequest(BaseModel):
    name: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("")
def get_watchlist():
    return {"watchlist": _load()}


@router.post("")
def add_to_watchlist(req: WatchlistAddRequest):
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    names = _load()
    if name.lower() in [n.lower() for n in names]:
        return {"message": f"'{name}' is already on the watchlist.", "watchlist": names}
    _add(name)
    return {"message": f"'{name}' added to watchlist.", "watchlist": _load()}


@router.delete("/{name}")
def remove_from_watchlist(name: str):
    removed = _remove(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"'{name}' not found in watchlist.")
    return {"message": f"'{name}' removed from watchlist.", "watchlist": _load()}


@router.get("/hits")
def get_watchlist_hits(limit: int = 50):
    """Return recent identification events that matched the watchlist."""
    from web.backend.services.audit_service import get_events
    events   = get_events(limit=1000, event_type="identify")
    watchlist = [n.lower() for n in _load()]
    hits      = [e for e in events if e.get("identity", "").lower() in watchlist]
    return {"hits": hits[:limit], "total": len(hits)}
