# Web Interface — Face Recognition UAV System

## Quick Start

```bat
cd "Face Recognition\web"
start.bat
```

Then open **http://localhost:8000** in your browser.

---

## Manual Start

From the **project root** (`Face Recognition/`):

```bash
# Install web dependencies
pip install fastapi uvicorn[standard] python-multipart aiofiles websockets

# Start the server
python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

| URL | Description |
|-----|-------------|
| `http://localhost:8000/` | Dashboard |
| `http://localhost:8000/gallery` | Enroll identities via images **or video files** (MP4/AVI/MOV/MKV) |
| `http://localhost:8000/identify` | Upload image → get prediction |
| `http://localhost:8000/demo` | Live webcam + video processing (adjustable frame skip) |
| `http://localhost:8000/results` | Experiment plots & metrics |
| `http://localhost:8000/analytics` | System analytics |
| `http://localhost:8000/audit` | Audit log |
| `http://localhost:8000/watchlist` | Alert watchlist |
| `http://localhost:8000/config` | Pipeline configuration |
| `http://localhost:8000/docs` | FastAPI auto-generated API docs |

---

## Prerequisites

- Python environment with `requirements.txt` installed
- Models downloaded via `python offline_setup.py` (dlib + InsightFace buffalo_l + UI fonts)
- MongoDB running locally (`mongod`) OR `MONGO_URI` in `.env` pointing to Atlas
- Trained baseline model in `results/baseline/models/`
  - Run `python experiments/run_baseline.py` first if not present
  - Or skip experiments and just use the web gallery to enroll + retrain

---

## Architecture

```
web/
├── backend/
│   ├── main.py                  ← FastAPI app
│   ├── config.py                ← Path configuration
│   ├── routers/
│   │   ├── gallery.py           ← Identity management
│   │   ├── identify.py          ← Face identification
│   │   ├── results.py           ← Experiment results
│   │   ├── experiments.py       ← Run experiments
│   │   ├── video_demo.py        ← WebSocket + video
│   │   └── config_router.py     ← Pipeline config
│   └── services/
│       ├── pipeline_service.py  ← Singleton pipeline
│       └── job_manager.py       ← Background jobs
└── frontend/
    ├── index.html               ← Dashboard
    ├── gallery.html
    ├── identify.html
    ├── results.html
    ├── demo.html
    ├── config.html
    ├── css/style.css            ← Self-hosted fonts (@font-face, no CDN after offline_setup.py)
    ├── fonts/                   ← Downloaded by offline_setup.py (Barlow/IBM Plex/Inter .ttf)
    └── js/
        ├── api.js
        ├── dashboard.js
        ├── gallery.js
        ├── identify.js
        ├── results.js
        ├── demo.js
        └── config.js
```
