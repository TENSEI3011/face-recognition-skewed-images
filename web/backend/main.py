"""
main.py - FastAPI application entrypoint

Starts up the pipeline on server boot, registers all routers,
and serves the frontend HTML/CSS/JS as static files.
"""

import sys
from pathlib import Path

# Fix Windows console encoding (cp1252 can't handle Unicode arrows/symbols
# used in docstrings/comments across the codebase). Must be set before any
# other imports that might trigger print() calls.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load .env file (local development); on Render env vars are set in dashboard
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from web.backend.config import FRONTEND_DIR
from web.backend.services import pipeline_service
from web.backend.routers import (
    gallery, identify, results, experiments, video_demo, config_router,
    auth, watchlist, audit, analytics, batch, retrain,
)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Face Recognition UAV System",
    description="Full-stack interface for the hybrid UAV face recognition pipeline.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow browser to call API from any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Load the face recognition pipeline and connect to MongoDB on startup."""
    import threading
    # Load ML pipeline in background thread (non-blocking)
    t = threading.Thread(target=pipeline_service.load_pipeline, daemon=True)
    t.start()
    # Connect to MongoDB Atlas (synchronous — fast, just a ping)
    from web.backend.services import mongo_service
    mongo_service.connect()


# ── Status endpoint ────────────────────────────────────────────────────────────
@app.get("/api/status")
def get_status():
    status = pipeline_service.get_status()
    gallery_count = 0
    gallery_dir = ROOT / "data" / "gallery"
    if gallery_dir.exists():
        gallery_count = len([d for d in gallery_dir.iterdir() if d.is_dir()])

    return {
        **status,
        "gallery_count": gallery_count,
        "server":        "running",
    }


# ── Include core routers ───────────────────────────────────────────────────────
app.include_router(gallery.router)
app.include_router(identify.router)
app.include_router(results.router)
app.include_router(experiments.router)
app.include_router(config_router.router)

# Video demo has mixed prefixes so include directly
app.include_router(video_demo.router)

# ── Include new feature routers ────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(watchlist.router)
app.include_router(audit.router)
app.include_router(analytics.router)
app.include_router(batch.router)
app.include_router(retrain.router)

# ── Frontend static files ──────────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    # Serve CSS/JS assets
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    app.mount("/js",  StaticFiles(directory=str(FRONTEND_DIR / "js")),  name="js")

    # Serve self-hosted fonts (downloaded by offline_setup.py)
    # Creates the directory if missing so the server never crashes even if
    # offline_setup.py hasn't been run yet — fonts simply fall back to CDN.
    fonts_dir = FRONTEND_DIR / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/fonts", StaticFiles(directory=str(fonts_dir)), name="fonts")


    # ── Original pages ──────────────────────────────────────────────────────────
    @app.get("/")
    def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/gallery")
    def serve_gallery():
        return FileResponse(str(FRONTEND_DIR / "gallery.html"))

    @app.get("/identify")
    def serve_identify():
        return FileResponse(str(FRONTEND_DIR / "identify.html"))

    @app.get("/results")
    def serve_results():
        return FileResponse(str(FRONTEND_DIR / "results.html"))

    @app.get("/demo")
    def serve_demo():
        return FileResponse(str(FRONTEND_DIR / "demo.html"))

    @app.get("/config")
    def serve_config():
        return FileResponse(str(FRONTEND_DIR / "config.html"))

    # ── New feature pages ───────────────────────────────────────────────────────
    @app.get("/login")
    def serve_login():
        return FileResponse(str(FRONTEND_DIR / "login.html"))

    @app.get("/watchlist")
    def serve_watchlist():
        return FileResponse(str(FRONTEND_DIR / "watchlist.html"))

    @app.get("/audit")
    def serve_audit():
        return FileResponse(str(FRONTEND_DIR / "audit.html"))

    @app.get("/analytics")
    def serve_analytics():
        return FileResponse(str(FRONTEND_DIR / "analytics.html"))

    @app.get("/batch")
    def serve_batch():
        return FileResponse(str(FRONTEND_DIR / "batch.html"))
