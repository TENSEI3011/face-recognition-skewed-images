"""
analytics.py — Analytics dashboard router

Derives statistics from the audit log.

Endpoints:
  GET /api/analytics/summary    → KPI cards (totals, known %, top identity)
  GET /api/analytics/daily      → per-day identification counts (last 30 days)
  GET /api/analytics/identities → per-identity hit counts
  GET /api/analytics/hourly     → per-hour activity heatmap
"""

from fastapi import APIRouter
from collections import defaultdict
from datetime import datetime, timezone, timedelta

from web.backend.services.audit_service import get_events, get_stats

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/summary")
def summary():
    return get_stats()


@router.get("/daily")
def daily_counts(days: int = 30):
    """Return per-day identification counts for the last `days` days."""
    events = get_events(limit=100_000, event_type="identify")
    counts: dict[str, int] = defaultdict(int)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    for e in events:
        ts_str = e.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.rstrip("Z")).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if ts >= cutoff:
            day = ts.strftime("%Y-%m-%d")
            counts[day] += 1

    # Fill in missing days with 0
    result = []
    for i in range(days):
        day = (datetime.now(timezone.utc) - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        result.append({"date": day, "count": counts.get(day, 0)})

    return {"daily": result}


@router.get("/identities")
def identity_counts():
    """Return per-identity recognition hit counts."""
    events = get_events(limit=100_000, event_type="identify")
    counts: dict[str, int] = defaultdict(int)
    known_counts: dict[str, int] = defaultdict(int)

    for e in events:
        ident = e.get("identity", "UNKNOWN")
        counts[ident] += 1
        if e.get("is_known"):
            known_counts[ident] += 1

    result = [
        {
            "identity": k,
            "total":    v,
            "known":    known_counts.get(k, 0),
        }
        for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)
    ]
    return {"identities": result}


@router.get("/hourly")
def hourly_heatmap():
    """Return per-hour activity counts (0–23) for the past 7 days."""
    events = get_events(limit=100_000, event_type="identify")
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    counts = [0] * 24

    for e in events:
        ts_str = e.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.rstrip("Z")).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if ts >= cutoff:
            counts[ts.hour] += 1

    return {"hourly": [{"hour": h, "count": c} for h, c in enumerate(counts)]}
