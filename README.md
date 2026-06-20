# Face Recognition using Skewed UAV Images

A hybrid face recognition system designed for **drone-based surveillance**, where faces are captured at oblique angles, varying altitudes, low resolution, and under motion blur.

## Pipeline

```
UAV Image
    ↓
MTCNN Face Detection
    ↓
Geometric Alignment → 112×112
    ↓        ↓         ↓         ↓
  HOG      LBP    Geometry   ArcFace
    ↓        ↓         ↓         ↓
         L2-Normalize + Concatenate
                    ↓
              PCA Reduction
                    ↓
               SVM Classifier
                    ↓
           Identity Prediction
```

## Features

- **HOG** — Histogram of Oriented Gradients (edge structure, illumination-robust)
- **LBP** — Local Binary Patterns (local texture, 8×8 spatial grid)
- **Facial Geometry** — dlib 68-point landmark ratios (illumination-invariant)
- **ArcFace** — InsightFace deep embedding, 512-D (pretrained on MS1MV3)
- **PCA** — Dimensionality reduction with 95% variance retention
- **SVM** — RBF kernel classifier with Platt probability calibration

## Project Structure

```
├── src/
│   ├── detection.py              ← MTCNN face detector
│   ├── alignment.py              ← Eye-corner alignment to 112×112
│   ├── augmentation.py           ← UAV degradation simulation (7 profiles)
│   ├── fusion.py                 ← L2 normalize + concatenate
│   ├── reducer.py                ← PCA with StandardScaler
│   ├── classifier.py             ← SVM (RBF, Platt-calibrated, GridSearchCV)
│   ├── pipeline.py               ← End-to-end orchestrator
│   └── features/
│       ├── hog_features.py
│       ├── lbp_features.py
│       ├── geometry_features.py
│       └── arcface_features.py
├── evaluation/
│   ├── metrics.py                ← Rank-k, EER, TAR@FAR, ROC, AUC, d-prime
│   └── visualizer.py             ← Publication-quality plots (dark theme)
├── experiments/
│   ├── run_baseline.py           ← Full pipeline experiment
│   ├── run_ablation.py           ← 10-combination ablation study
│   ├── run_pose_study.py         ← Pose / altitude stratified evaluation
│   └── run_degradation.py        ← Degradation sweep (7 profiles + altitude)
├── data/
│   ├── gallery/                  ← Enrollment images (per-identity folders)
│   └── probe/                    ← Test/query images (per-identity folders)
├── models/                       ← Saved PCA + SVM + dlib models
├── results/                      ← Experiment outputs (plots, JSON)
├── download_lfw.py               ← LFW dataset downloader & organizer
├── setup.py                      ← Project setup + dlib model download
└── requirements.txt
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Setup (downloads dlib 68-point landmark model)
```bash
python setup.py
```

### 3. Download a Dataset

**Option A — LFW (quickest start):**
```bash
# Download lfw.tgz from http://vis-www.cs.umass.edu/lfw/
# Place it in data/datasets/lfw.tgz, then:
python download_lfw.py
```

**Option B — Manual:**
```
data/gallery/<identity_name>/*.jpg   ← enrollment images
data/probe/<identity_name>/*.jpg     ← query images
```

### 4. Run Experiments
```bash
# Full pipeline baseline
python experiments/run_baseline.py

# Ablation study (HOG vs LBP vs ArcFace vs all combinations)
python experiments/run_ablation.py

# Pose-stratified evaluation (per yaw/pitch/altitude bin)
python experiments/run_pose_study.py

# Degradation sweep (clean → extreme, altitude 5m → 30m)
python experiments/run_degradation.py
```

## UAV Degradation Profiles

The `UAVAugmentor` in `src/augmentation.py` simulates real drone imaging conditions:

| Profile | Simulates | Blur | Noise | JPEG |
|---------|-----------|------|-------|------|
| `CLEAN` | Stable hover, 5m | None | None | None |
| `MILD` | 5–10m altitude | Low | Low | Low |
| `MODERATE` | 10–20m altitude | Medium | Medium | Medium |
| `SEVERE` | 20–30m altitude | High | High | High |
| `EXTREME` | >30m altitude | Very High | Very High | Very High |
| `MOTION` | Moving drone | Directional | Low | — |
| `COMBINED` | Worst case | Mixed | High | High |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Rank-1 IR** | Top-1 identification accuracy (CMC) |
| **Rank-5 IR** | Top-5 identification accuracy |
| **EER** | Equal Error Rate (FAR = FRR) |
| **TAR @ FAR=0.1%** | True Accept Rate at 0.1% False Accepts (NIST) |
| **AUC** | Area under ROC Curve |
| **d' (d-prime)** | Signal detection discriminability index |

## Ablation Study

Runs all 10 modality combinations automatically:

| Configuration | Description |
|---|---|
| HOG only | Gradient-based baseline |
| LBP only | Texture-based baseline |
| Geometry only | Landmark ratio baseline |
| ArcFace only | Deep embedding baseline |
| HOG + LBP | Classical fusion |
| HOG + ArcFace | Gradient + deep |
| LBP + ArcFace | Texture + deep |
| HOG + LBP + Geometry | Full classical |
| HOG + LBP + ArcFace | Classical + deep (no geometry) |
| **All (Full Pipeline)** | **Best expected performance** |

## Recommended Datasets

| Dataset | Purpose | Link |
|---------|---------|------|
| LFW | Ground-level baseline | [vis-www.cs.umass.edu/lfw](http://vis-www.cs.umass.edu/lfw/) |
| CFP | Frontal vs. profile (pose study) | [cfpw.io](http://www.cfpw.io/) |
| DroneSURF | UAV-specific (primary) | [iab-rubric.org](https://iab-rubric.org/resources/dronesurf) |
| UAV-Human | Multi-altitude UAV | [GitHub](https://github.com/SUTDCV/UAV-Human) |
| SCface | Low-res surveillance proxy | [scface.org](http://www.scface.org/) |
| TinyFace | Extreme low-resolution | [cs.cmu.edu](https://cs.cmu.edu/~peiyunh/tiny/) |

## References

1. Dalal & Triggs (2005). *Histograms of Oriented Gradients for Human Detection.* CVPR.
2. Ahonen et al. (2006). *Face Description with Local Binary Patterns.* IEEE TPAMI.
3. Deng et al. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.* CVPR.
4. Zhang et al. (2016). *Joint Face Detection and Alignment Using MTCNN.* IEEE SPL.
5. King (2009). *dlib-ml: A Machine Learning Toolkit.* JMLR.
6. Turk & Pentland (1991). *Eigenfaces for Recognition.* J. Cognitive Neuroscience.

## License

MIT License — see [LICENSE](LICENSE) for details.
