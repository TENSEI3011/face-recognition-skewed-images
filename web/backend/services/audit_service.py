"""
audit_service.py — Audit log service

PRIMARY:  MongoDB Atlas  (audit_log collection)
FALLBACK: web/audit_log.jsonl (if MongoDB is not connected)

Every identification call logs a JSON event.
Provides read, search, CSV export, and PDF export.
"""

import json
import csv
import io
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── Local fallback log file ────────────────────────────────────────────────────
_LOG_FILE = Path(__file__).resolve().parents[2] / "audit_log.jsonl"


def _ts() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# ── Write ─────────────────────────────────────────────────────────────────────

def log_event(event_type: str, data: dict) -> None:
    """Append a single JSON event to MongoDB or local file."""
    entry = {
        "timestamp": _ts(),
        "event":     event_type,
        **data,
    }

    # Try MongoDB first
    try:
        from web.backend.services.mongo_service import get_db
        db = get_db()
        if db is not None:
            db["audit_log"].insert_one({**entry})
            return
    except Exception:
        pass

    # Fallback: local .jsonl file
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ── Read ──────────────────────────────────────────────────────────────────────

def get_events(limit: int = 200, event_type: Optional[str] = None,
               identity: Optional[str] = None) -> list[dict]:
    """Return up to `limit` most-recent events, optionally filtered."""

    # Try MongoDB first
    try:
        from web.backend.services.mongo_service import get_db
        db = get_db()
        if db is not None:
            query = {}
            if event_type:
                query["event"] = event_type
            if identity:
                query["identity"] = {"$regex": f"^{identity}$", "$options": "i"}
            cursor = db["audit_log"].find(
                query, {"_id": 0}
            ).sort("timestamp", -1).limit(limit)
            return list(cursor)
    except Exception:
        pass

    # Fallback: local .jsonl
    if not _LOG_FILE.exists():
        return []
    lines = _LOG_FILE.read_text(encoding="utf-8").strip().splitlines()
    events = []
    for line in reversed(lines):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if event_type and e.get("event") != event_type:
            continue
        if identity and e.get("identity", "").lower() != identity.lower():
            continue
        events.append(e)
        if len(events) >= limit:
            break
    return events


def get_stats() -> dict:
    """Return aggregate statistics derived from the audit log."""
    events = get_events(limit=100_000)
    ident_events = [e for e in events if e.get("event") == "identify"]
    total = len(ident_events)
    known = sum(1 for e in ident_events if e.get("is_known"))
    identity_counts: dict[str, int] = {}
    for e in ident_events:
        ident = e.get("identity", "UNKNOWN")
        identity_counts[ident] = identity_counts.get(ident, 0) + 1
    top = sorted(identity_counts.items(), key=lambda x: x[1], reverse=True)
    return {
        "total_identifications": total,
        "known_count":           known,
        "unknown_count":         total - known,
        "known_pct":             round(known / total * 100, 1) if total else 0.0,
        "top_identities":        [{"identity": k, "count": v} for k, v in top[:10]],
    }


# ── CSV Export ────────────────────────────────────────────────────────────────

def export_csv(events: list[dict]) -> bytes:
    """Return CSV bytes for the given event list."""
    if not events:
        return b"timestamp,event,identity,confidence,is_known,source\n"
    buf = io.StringIO()
    fieldnames = ["timestamp", "event", "identity", "confidence", "is_known", "source", "filename"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(events)
    return buf.getvalue().encode("utf-8")


# ── PDF Export ────────────────────────────────────────────────────────────────

def export_pdf(events: list[dict]) -> bytes:
    """Generate a PDF audit report using reportlab. Falls back to CSV if unavailable."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.units import cm

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        story.append(Paragraph("Face Recognition UAV System — Audit Report", styles["Title"]))
        story.append(Paragraph(f"Generated: {_ts()}", styles["Normal"]))
        story.append(Spacer(1, 0.5*cm))

        stats = get_stats()
        story.append(Paragraph(
            f"Total Identifications: <b>{stats['total_identifications']}</b> | "
            f"Known: <b>{stats['known_count']}</b> | "
            f"Unknown: <b>{stats['unknown_count']}</b>",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.5*cm))

        headers = ["Timestamp", "Event", "Identity", "Confidence", "Known?", "Source"]
        rows    = [headers]
        for e in events[:500]:
            rows.append([
                e.get("timestamp", "")[:19],
                e.get("event", ""),
                e.get("identity", "—"),
                f"{float(e.get('confidence', 0))*100:.1f}%" if e.get("confidence") else "—",
                "✓" if e.get("is_known") else "✗",
                e.get("source", "—"),
            ])

        t = Table(rows, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2744")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTSIZE",   (0, 0), (-1, 0), 9),
            ("FONTSIZE",   (0, 1), (-1, -1), 7.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f2f7")]),
            ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#e2e8f0")),
            ("ALIGN",      (3, 1), (4, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
        doc.build(story)
        return buf.getvalue()

    except ImportError:
        return export_csv(events)
