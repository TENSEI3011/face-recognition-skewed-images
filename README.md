---
title: Face Recognition UAV System
emoji: 🎭
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: true
app_port: 7860
---

# Face Recognition on Skewed UAV Images

A **multi-modal face recognition system** designed for drone-based surveillance, where faces are captured at oblique angles, varying altitudes, low resolution, and under motion blur. Built with a modern web interface for real-time identification, gallery management, video processing, and experiment analysis.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![InsightFace](https://img.shields.io/badge/InsightFace-buffalo__l-orange.svg)](https://github.com/deepinsight/insightface)

---

## Pipeline

```
UAV Image / Video Frame
        ↓
SCRFD Face Detection (det_10g.onnx)     ← full-res, auto-upscale for tiny faces
        ↓
dlib + InsightFace 5-pt Alignment → 112×112
        ↓
  🛡️ Passive Liveness Check (anti-spoofing)   ← rejects printed photos, screens, video replay
    score < 0.45 → SPOOF DETECTED (stop)       ← threshold lowered to 0.45 (6-signal fusion)
    score ≥ 0.45 → continue
        ↓
  🔍 Active Blink Challenge (live webcam only)  ← browser-side, no server round-trip
    no blink in 10s → rejected
    blink detected → WebSocket opens
        ↓
     ┌──────────────────────────────┐  (parallel — ThreadPoolExecutor)
     ↓          ↓           ↓           ↓
    HOG         LBP      Geometry    ArcFace (512-D)
     ↓          ↓           ↓           ↓
          L2-Normalize + Concatenate
                      ↓
                PCA (99% variance)           ← raised from 95% for better accuracy
                      ↓
       SVM Classifier (RBF, Platt-calibrated)
                      ↓
            TTA — 5-variant embedding average  ← NEW: +3–8% accuracy on degraded images
                      ↓
     FAISS Open-Set Cosine Matcher (IndexFlatIP)
                      ↓
         Identity / UNKNOWN (score < FAISS_THRESHOLD)
```

---

## Key Features

| Component | Details |
|---|---|
| **Face Detector** | SCRFD `det_10g.onnx` (InsightFace `buffalo_l`) — replaces MTCNN, 5× faster, 93% WiderFace mAP |
| **Deep Embedding** | ArcFace ResNet-50 (`w600k_r50.onnx`) — 512-D cosine embedding |
| **HOG** | Histogram of Oriented Gradients — edge/gradient structure, illumination-robust |
| **LBP** | Local Binary Patterns — local texture, rotation and illumination invariant |
| **Geometry** | dlib 68-landmark inter-distance ratios — pose-stable structural descriptor |
| **Parallel Extraction** | `ThreadPoolExecutor` runs HOG/LBP/Geometry/ArcFace concurrently — faster inference |
| **Embedding Cache** | MD5-hash based `.npy` cache — retrains 10× faster after first run |
| **PCA** | Dimensionality reduction with **99%** variance retention (raised from 95%) |
| **SVM** | RBF kernel with Platt probability calibration — closed-set fallback |
| **TTA** | Test-Time Augmentation — 5-variant embedding average before FAISS matching (+3–8% accuracy) |
| **FAISS** | `IndexFlatIP` cosine similarity — open-set identification with UNKNOWN rejection (`FAISS_THRESHOLD=0.35`) |
| **🛡️ Passive Liveness** | LBP texture entropy + FFT frequency + gradient coherence (6-signal fusion) — rejects photo/screen attacks in ~3ms |
| **🔍 Active Blink Challenge** | Browser-side pixel brightness analysis detects real blink within 10s; static photos/screens cannot blink |
| **Temporal Voter + Identity Tracker** | Dual-strategy tracker: identity-name lookup + IoU spatial fallback; holds labels 4s through occlusion/blur |
| **Fast-Confirm** | High-confidence frames (≥0.42 live / ≥0.60 upload) bypass voter for instant identification |
| **EXIF-Aware Decode** | Pillow `exif_transpose` corrects phone selfie orientation before detection |
| **Web Interface** | FastAPI + vanilla JS dashboard with gallery, identification, video, experiments, audit log |
| **RBAC** | JWT authentication with role-based access control |

---

## Recent Updates

### 🆕 Latest (July 2026)

- **🔍 Active blink liveness challenge** — `demo.js` now runs a fully browser-side blink detection challenge before opening the WebSocket stream. Pixel-level eye-strip brightness analysis (~12fps) detects a real blink (>2 consecutive dark frames) within a 10-second window. A printed photo or screen replay cannot blink — defeating presentation attacks without any server round-trip.
- **🎯 Improved temporal identity tracking** — `video_demo.py` WebSocket stream now uses a dual-strategy tracker: **identity-name lookup** (primary, follows moving persons) + **spatial IoU fallback** (holds label when FAISS returns UNKNOWN due to blur or occlusion). Confirmed identities are held for 4 seconds (`HOLD_S=4.0`) so labels no longer flicker when subjects turn their heads.
- **⚡ Fast-confirm threshold** — High-confidence frames (score ≥ 0.42 for live stream, ≥ 0.60 for video upload) now bypass the temporal voter and instantly confirm identity — eliminating the "UNKNOWN at start" delay for clear frontal views.
- **📦 Mummy gallery enrolled** — New identity `mummy` added to `data/gallery/mummy/` from 22 video frames extracted via `extract_gallery_frames.py`. Embedding cache files (`.npy`/`.hash`) are pre-generated for instant retrain.
- **🛠 Updated thresholds in `config.py`** — `FAISS_THRESHOLD` tuned to `0.35` for moving multi-person live cam (was 0.38); `LIVENESS_THRESHOLD` lowered to `0.45` (was 0.50) with 6-signal passive fusion; `DEFAULT_PCA_VARIANCE` raised to `0.99`; active blink config added (`BLINK_ENABLED`, `BLINK_EAR_THRESHOLD`, `BLINK_TIMEOUT_SEC`).
- **🔬 EXIF-aware image decode** — `identify.py` uses Pillow `ImageOps.exif_transpose` before OpenCV decode so phone selfies (iOS/Android) are correctly oriented without manual rotation.
- **🎥 Updated PDF guides** — `Face_Recognition_Pipeline_Explanation.pdf`, `Setup_Guide_Face_Recognition.pdf`, and `Army_Offline_Deployment_Guide.pdf` regenerated with latest architecture details, threshold tables, and deployment instructions.
- **🔤 Unicode fix utility** — `fix_unicode.py` utility added to repair Windows CRLF / encoding issues in generated PDF scripts.

### Previous Updates

- **🛡️ Passive liveness detection (anti-spoofing)** — `src/liveness.py` PassiveLivenessDetector uses LBP texture entropy, FFT frequency peak analysis, and gradient coherence scoring to reject presentation attacks (~3ms). Integrated into `/identify`, `/identify/frame`, and live WebSocket stream.
- **Embedding cache** — `src/embedding_cache.py` caches feature vectors by MD5 hash; retrains drop from ~15s to ~2s after first run (10× faster).
- **Parallel feature extraction** — `pipeline.py` uses `ThreadPoolExecutor` to run HOG, LBP, Geometry, and ArcFace concurrently on each face crop.
- **PCA variance raised to 99%** — retains more discriminative components; +2–5% accuracy on degraded UAV images.
- **Test-Time Augmentation (TTA)** — `identify.py` averages embeddings over 5 augmented variants before FAISS matching (+3–8% accuracy on compressed/blurry frames).
- **Incremental retrain** — embedding cache means only new/changed gallery images are re-processed.
- **Offline / air-gapped deployment** — `offline_setup.py` pre-downloads all internet assets; UI fonts are self-hosted.
- **Local MongoDB support** — `.env` supports Atlas / local Community Edition / disk-only fallback.
- **Video enrollment** — Gallery accepts MP4/AVI/MOV/MKV; frames sampled at configurable interval.
- **Full-resolution video detection** — SCRFD receives full-res frames so 17–30px faces at UAV altitude are not lost.
- **FAISS open-set matching** — replaced SVM-only ID with FAISS cosine search + threshold.
- **bcrypt authentication** — salted bcrypt replaces raw SHA-256.

---

## Project Structure

```
face-recognition-skewed-images/
├── src/
│   ├── detection.py              ← SCRFD face detector (replaces MTCNN)
│   ├── liveness.py               ← NEW: passive anti-spoofing detector
│   ├── embedding_cache.py        ← NEW: MD5 hash-based feature cache (10× faster retrains)
│   ├── matcher.py                ← FAISS open-set cosine matcher
│   ├── classifier.py             ← SVM (RBF, Platt-calibrated, closed-set fallback)
│   ├── pipeline.py               ← Parallel feature extraction (ThreadPoolExecutor)
│   ├── alignment.py              ← 5-point similarity transform to 112×112
│   ├── augmentation.py           ← UAV degradation simulation (7 profiles)
│   ├── fusion.py                 ← L2 normalize + concatenate features
│   ├── reducer.py                ← PCA with StandardScaler (99% variance)
│   └── features/
│       ├── hog_features.py
│       ├── lbp_features.py
│       ├── geometry_features.py
│       └── arcface_features.py
├── web/
│   ├── backend/
│   │   ├── main.py               ← FastAPI app entry point
│   │   ├── config.py             ← Thresholds: FAISS=0.38, LIVENESS=0.50, PCA=0.99
│   │   ├── routers/
│   │   │   ├── identify.py       ← /api/identify (liveness gate + TTA + FAISS)
│   │   │   ├── gallery.py        ← /api/gallery  (enroll, retrain, delete)
│   │   │   ├── video_demo.py     ← /api/video + WebSocket (liveness per-frame)
│   │   │   ├── config_router.py  ← /api/config   (liveness + threshold settings)
│   │   │   ├── results.py        ← /api/results  (experiment plots)
│   │   │   ├── experiments.py    ← /api/experiments/run/{exp}
│   │   │   ├── auth.py           ← JWT + bcrypt authentication
│   │   │   ├── retrain.py        ← /api/retrain (cache-first background job)
│   │   │   ├── batch.py          ← /api/batch
│   │   │   ├── analytics.py      ← /api/analytics
│   │   │   ├── watchlist.py      ← /api/watchlist
│   │   │   └── audit.py          ← /api/audit (logs liveness_score + is_spoof)
│   │   └── services/
│   │       ├── pipeline_service.py  ← Singleton pipeline + FAISS loader
│   │       ├── liveness_service.py  ← NEW: singleton liveness detector wrapper
│   │       ├── temporal_service.py  ← TemporalVoter (10-frame rolling window)
│   │       ├── enhance_service.py   ← Image enhancement / quality service
│   │       ├── auth_service.py      ← bcrypt user management
│   │       └── job_manager.py       ← Background experiment jobs
│   └── frontend/
│       ├── index.html / identify.html / gallery.html / ...
│       ├── css/style.css / tour.css
│       └── js/api.js / identify.js / gallery.js / results.js / demo.js / tour.js / config.js
├── experiments/
│   ├── run_baseline.py           ← Full pipeline evaluation (CMC, ROC, EER, AUC, d')
│   ├── run_ablation.py           ← 10-modality combination ablation study
│   ├── run_pose_study.py         ← Pose/altitude stratified evaluation
│   └── run_degradation.py        ← Degradation sweep (CLEAN → EXTREME)
├── evaluation/
│   ├── metrics.py                ← Rank-k, EER, TAR@FAR, AUC, d-prime
│   └── visualizer.py             ← Publication-quality plots
├── data/
│   ├── gallery/                  ← Enrollment images (one subfolder per identity)
│   └── probe/                    ← Test/query images
├── models/                       ← dlib landmark model (downloaded by offline_setup.py)
├── results/                      ← Experiment outputs (plots + JSON metrics)
├── offline_setup.py              ← One-shot offline asset downloader
├── extract_gallery_frames.py     ← Utility to extract face crops from a video
├── setup.py                      ← Auto-downloads dlib model
└── requirements.txt
```

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/TENSEI3011/face-recognition-skewed-images.git
cd face-recognition-skewed-images
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

> **Python 3.10 or 3.11 required.** Python 3.13 breaks `dlib-bin`.

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `dlib-bin` fails on Windows:
```bash
pip install cmake && pip install dlib
pip install -r requirements.txt
```

### 4. Download Required Models (One-Time Internet Setup)

**Recommended — downloads everything at once:**
```bash
python offline_setup.py
```
This downloads:
- dlib 68-point landmark model (~100 MB) → `models/`
- InsightFace `buffalo_l` (ArcFace + SCRFD, ~500 MB) → `~/.insightface/models/buffalo_l/`
- All 9 UI fonts → `web/frontend/fonts/`

After this, the system runs **fully offline** — no internet required.

**Alternative — models only:**
```bash
python setup.py          # dlib only
# InsightFace downloads automatically on first run
```

### 5. Configure Environment
```bash
copy .env.example .env      # Windows
cp .env.example .env        # Linux / macOS
```

Edit `.env` — choose ONE MongoDB option:
```env
JWT_SECRET_KEY=your-random-secret-key-here

# Option A: Local MongoDB (recommended for offline/defence)
MONGO_URI=mongodb://localhost:27017

# Option B: MongoDB Atlas (cloud, requires internet)
# MONGO_URI=mongodb+srv://user:password@cluster.mongodb.net/

# Option C: No MongoDB (disk-only fallback — no audit log or watchlist)
# Leave MONGO_URI commented out

MONGO_DB_NAME=facerecog_db
ENV=development
```

> **Local MongoDB**: Install [MongoDB Community Server](https://www.mongodb.com/try/download/community) and start with `mongod --dbpath C:\data\db` (Windows) or `mongod --dbpath /data/db` (Linux).

### 6. Add Gallery Images

```
data/gallery/
    person_a/
        photo1.jpg
        photo2.jpg
    person_b/
        photo1.jpg
```

Or use the web gallery page to upload via the browser UI.

### 7. Run the Web Interface
```bash
python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

---

## Web Interface Pages

| URL | Description |
|---|---|
| `http://localhost:8000/` | Dashboard — system status, metrics overview |
| `http://localhost:8000/identify` | Upload image → get face identified with ranked candidates |
| `http://localhost:8000/gallery` | Enroll identities, view gallery, trigger retrain |
| `http://localhost:8000/demo` | Live webcam stream + video file upload processing |
| `http://localhost:8000/results` | Experiment plots (CMC, ROC, confusion matrix) |
| `http://localhost:8000/config` | Adjust FAISS threshold, blur gates, pipeline settings |
| `http://localhost:8000/analytics` | System usage analytics |
| `http://localhost:8000/audit` | Audit log of all identification events |
| `http://localhost:8000/watchlist` | Alert watchlist management |
| `http://localhost:8000/docs` | FastAPI auto-generated REST API docs |

---

## Run Experiments
```bash
# Full pipeline baseline (CMC, ROC, EER, TAR@FAR, AUC, d-prime)
python experiments/run_baseline.py

# Ablation study — all 10 modality combinations
python experiments/run_ablation.py

# Pose-stratified evaluation (yaw / pitch / altitude bins)
python experiments/run_pose_study.py

# Degradation sweep (CLEAN → EXTREME, altitude 5m → 30m)
python experiments/run_degradation.py
```

Results (plots + JSON) are saved to `results/<experiment_name>/`.
View them at **http://localhost:8000/results**.

### Retrain with GridSearchCV (accuracy boost)
After finalizing the gallery, run a GridSearch retrain to find optimal SVM hyperparameters:
```bash
curl -X POST "http://localhost:8000/api/pipeline/retrain?use_grid_search=true"
```
Or via the **API Docs** page at `/docs`. Takes ~5 minutes, improves Rank-1 by 5–15%.

---

## Enroll a New Person

### From Images (via web UI)
Open **http://localhost:8000/gallery**, enter an identity name, and upload photos or a short video clip.

### From a Video File (via web UI)
On the Gallery page, upload an MP4/MOV/AVI file. Frames are sampled every N frames (default 15) and each detected face is embedded individually. A 20-second 30fps video at interval=15 yields ~40 enrolled embeddings.

### From the Command Line
```bash
python extract_gallery_frames.py --video "person.mp4" --name "person_name" --count 20
```
Extracts 20 high-quality face crops and saves them to `data/gallery/person_name/`.
Then go to the web gallery to trigger a retrain.

---

## UAV Degradation Profiles

| Profile | Simulates | Blur | Noise | JPEG |
|---|---|---|---|---|
| `CLEAN` | Stable hover, 5m | None | None | None |
| `MILD` | 5–10m altitude | Low | Low | Low |
| `MODERATE` | 10–20m altitude | Medium | Medium | Medium |
| `SEVERE` | 20–30m altitude | High | High | High |
| `EXTREME` | >30m altitude | Very High | Very High | Very High |
| `MOTION` | Moving drone | Directional | Low | — |
| `COMBINED` | Worst case | Mixed | High | High |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Rank-1 IR** | Top-1 identification accuracy (CMC curve) |
| **Rank-5 IR** | Top-5 identification accuracy |
| **EER** | Equal Error Rate — threshold where FAR = FRR |
| **TAR @ FAR=0.1%** | True Accept Rate at 0.1% False Accept Rate |
| **AUC** | Area under ROC Curve |
| **d' (d-prime)** | Signal detection discriminability index |

---

## Open-Set Configuration

The FAISS cosine threshold controls the boundary between known and unknown identities:

```python
# web/backend/config.py
FAISS_THRESHOLD      = 0.35   # Tuned for moving multi-person live cam
                               # (was 0.38 — lowered for motion-blurred faces)
LIVENESS_ENABLED     = True   # anti-spoofing: reject presentation attacks
LIVENESS_THRESHOLD   = 0.45  # Lowered from 0.50 — passive is now 6-signal fused

# ── Active Liveness — Blink Challenge ──────────────────────────────────────
BLINK_ENABLED        = True   # Set False to skip blink challenge (testing only)
BLINK_EAR_THRESHOLD  = 0.25   # Eye Aspect Ratio below this → eye closed (blink)
BLINK_TIMEOUT_SEC    = 7.0    # Seconds user has to complete the blink challenge
BLINK_REQUIRED_COUNT = 1      # Number of blinks required to pass
BLINK_CONSEC_FRAMES  = 2      # Min consecutive frames below EAR threshold = blink
```

| Threshold | Effect |
|---|---|
| Higher (0.55+) | Fewer false accepts, more UNKNOWN results |
| Lower (0.25–0.35) | More accepts, better for degraded/compressed video |
| Default (0.35) | Tuned for moving multi-person live cam with motion blur |

### Liveness / Anti-Spoofing

| Setting | Effect |
|---|---|
| `LIVENESS_ENABLED=True` | Reject faces scoring below threshold |
| `LIVENESS_THRESHOLD=0.45` | Default: 6-signal fusion, balanced for outdoor UAV |
| `LIVENESS_THRESHOLD=0.60` | Stricter: better for controlled indoor checkpoints |
| `LIVENESS_THRESHOLD=0.35` | Lenient: better for highly degraded/compressed video |

### Active Blink Challenge (Live Webcam)

| Setting | Effect |
|---|---|
| `BLINK_ENABLED=True` | Require a real blink before WebSocket recognition opens |
| `BLINK_TIMEOUT_SEC=7.0` | 7 seconds for the user to blink (browser-side timer) |
| `BLINK_CONSEC_FRAMES=2` | Min 2 consecutive dark eye-strip frames counts as a blink |
| `skipLivenessChallenge()` | JS function to bypass in demo/testing mode |

---

## Requirements

```
Python         3.10 or 3.11
fastapi        >=0.104
uvicorn        >=0.24
insightface    >=0.7.3     (SCRFD + ArcFace)
onnxruntime    >=1.16
dlib-bin       >=19.24
faiss-cpu      >=1.7.4     (open-set FAISS matching)
opencv-python-headless >=4.8
scikit-learn   >=1.3
passlib[bcrypt] >=1.7.4   (secure password hashing)
pymongo        >=4.6       (optional — MongoDB integration)
```

---

## References

1. Guo et al. (2021). *Sample and Computation Redistribution for Efficient Face Detection.* ICLR 2022. (SCRFD)
2. Deng et al. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR.
3. Johnson et al. (2019). *Billion-scale Similarity Search with GPUs.* IEEE Trans. Big Data. (FAISS)
4. Dalal & Triggs (2005). *Histograms of Oriented Gradients for Human Detection.* CVPR.
5. Ahonen et al. (2006). *Face Description with Local Binary Patterns.* IEEE TPAMI.
6. Kazemi & Sullivan (2014). *One Millisecond Face Alignment with an Ensemble of Regression Trees.* CVPR.
7. Cortes & Vapnik (1995). *Support-Vector Networks.* Machine Learning.
8. Bendale & Boult (2015). *Towards Open Set Deep Networks.* CVPR.
9. Turk & Pentland (1991). *Eigenfaces for Recognition.* J. Cognitive Neuroscience.
10. Boulkenafet et al. (2017). *Face Anti-Spoofing Using Texture Analysis.* IEEE TIFS. (LBP-based PAD)
11. Li et al. (2004). *Live Face Detection Based on the Analysis of Fourier Spectra.* SPIE.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
