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

## Pages

| URL | Description |
|-----|-------------|
| `http://localhost:8000/` | Dashboard |
| `http://localhost:8000/gallery` | Enroll & manage identities |
| `http://localhost:8000/identify` | Upload image → get prediction |
| `http://localhost:8000/demo` | Live webcam + video processing |
| `http://localhost:8000/results` | Experiment plots & metrics |
| `http://localhost:8000/config` | Pipeline configuration |
| `http://localhost:8000/docs` | FastAPI auto-generated API docs |

---

## Prerequisites

- Python environment with `requirements.txt` installed
- Trained baseline model in `results/baseline/models/`
  - Run `python experiments/run_baseline.py` first if not present
- dlib model at `models/shape_predictor_68_face_landmarks.dat`
- InsightFace ArcFace model (auto-downloaded on first use)

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
    ├── css/style.css
    └── js/
        ├── api.js
        ├── dashboard.js
        ├── gallery.js
        ├── identify.js
        ├── results.js
        ├── demo.js
        └── config.js
```
