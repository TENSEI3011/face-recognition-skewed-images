# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Spaces — Face Recognition UAV System
# Base: Python 3.11 slim (smaller image, faster build)
# Port: 7860 (HuggingFace Spaces default)
# RAM:  16 GB free tier (plenty for buffalo_l ArcFace + all models)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# libGL / libGLU      → OpenCV needs these for display functions
# libglib2.0-0        → GLib (required by OpenCV internals)
# libgomp1            → OpenMP (parallel processing for dlib, onnxruntime)
# libsm6 libxext6     → X11 libs (OpenCV import requires even without display)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user (HuggingFace Spaces requirement) ─────────────────────
RUN useradd -m -u 1000 user
USER user

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKDIR /home/user/app
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
# Store InsightFace models in the app directory (writable, inside workdir)
ENV INSIGHTFACE_HOME=/home/user/.insightface
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Install Python packages ───────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ── Copy project source ───────────────────────────────────────────────────────
COPY --chown=user:user . .

# ── Download InsightFace models at BUILD TIME ──────────────────────────────────
# WHY AT BUILD TIME:
#   HuggingFace Spaces has a slow cold-start if models are downloaded at runtime.
#   Baking models into the Docker image means the server starts instantly.
#   buffalo_l downloads: det_10g.onnx (~16MB) + w600k_r50.onnx (~280MB)
RUN python -c "
import os
os.environ['INSIGHTFACE_HOME'] = '/home/user/.insightface'
from insightface.app import FaceAnalysis

# Download detection model (SCRFD det_10g, 16MB)
det = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'], providers=['CPUExecutionProvider'])
det.prepare(ctx_id=-1, det_size=(640, 640))
print('[Docker] Detection model (buffalo_l) downloaded OK')

# Download full recognition model (ArcFace ResNet-100, 280MB)
rec = FaceAnalysis(name='buffalo_l', allowed_modules=['detection','recognition'], providers=['CPUExecutionProvider'])
rec.prepare(ctx_id=-1, det_size=(640, 640))
print('[Docker] ArcFace recognition model (buffalo_l) downloaded OK')
print('[Docker] All InsightFace models ready!')
"

# ── Expose HuggingFace default port ──────────────────────────────────────────
EXPOSE 7860

# ── Start FastAPI server ──────────────────────────────────────────────────────
CMD ["uvicorn", "web.backend.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
