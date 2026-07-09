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

A full-stack face recognition system designed for UAV/drone surveillance footage.

## Features

- **Live Webcam Demo** — Real-time face identification via WebSocket
- **Gallery Management** — Upload face photos, embeddings stored in MongoDB Atlas
- **Batch Processing** — Identify faces in uploaded videos/images
- **Audit Log** — Full history of all identifications
- **Watchlist Alerts** — Get notified when a specific person is detected
- **UAV Degradation Simulation** — Test recognition under 7 levels of image degradation

## Tech Stack

- **Backend**: FastAPI + Python 3.11
- **Face Detection**: InsightFace SCRFD (93% mAP on WiderFace)
- **Face Recognition**: ArcFace ResNet-100 (buffalo_l, 512-D embeddings)
- **Database**: MongoDB Atlas (free tier, persistent storage)
- **Frontend**: Vanilla HTML/CSS/JS

## API

The app runs on port 7860. Key endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Frontend dashboard |
| `/api/gallery/upload` | POST | Upload face photos |
| `/api/gallery` | GET | List enrolled identities |
| `/ws/stream` | WS | Live webcam recognition stream |
| `/api/identify` | POST | Identify a face image |
| `/api/audit` | GET | View audit log |
| `/api/watchlist` | GET/POST/DELETE | Manage watchlist |

## Environment Variables

Set these in **Space Settings → Repository Secrets**:

| Variable | Description |
|---|---|
| `MONGO_URI` | MongoDB Atlas connection string |
| `MONGO_DB_NAME` | Database name (default: `facerecog_db`) |
