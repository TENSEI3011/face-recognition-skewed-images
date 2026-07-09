"""
audit.py — Audit log router

Endpoints:
  GET /api/audit               → recent events (paginated, filterable)
  GET /api/audit/stats         → aggregate statistics
  GET /api/audit/export/csv    → download CSV
  GET /api/audit/export/pdf    → download PDF report
"""

from fastapi import APIRouter, Query
from fastapi.responses import Response
from typing import Optional

from web.backend.services.audit_service import (
    get_events, get_stats, export_csv, export_pdf
)

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("")
def list_events(
    limit:      int            = Query(200, ge=1, le=1000),
    event_type: Optional[str]  = Query(None),
    identity:   Optional[str]  = Query(None),
):
    events = get_events(limit=limit, event_type=event_type, identity=identity)
    return {"events": events, "count": len(events)}


@router.get("/stats")
def audit_stats():
    return get_stats()


@router.get("/export/csv")
def export_csv_route(limit: int = Query(1000, ge=1, le=10000)):
    events = get_events(limit=limit)
    csv_bytes = export_csv(events)
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit_log.csv"},
    )


@router.get("/export/pdf")
def export_pdf_route(limit: int = Query(500, ge=1, le=1000)):
    events = get_events(limit=limit)
    pdf_bytes = export_pdf(events)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=audit_report.pdf"},
    )
