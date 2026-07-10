"""
seed_gallery.py - Preload local gallery images into MongoDB Atlas

Reads all images from data/gallery/{identity}/ folders,
extracts ArcFace embeddings, and stores them in MongoDB.

Run ONCE before sharing the live URL with scientists:
    python seed_gallery.py
"""

import sys
import os

# Fix Windows terminal encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# Load .env
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import cv2
import base64
import numpy as np

GALLERY_DIR = ROOT / "data" / "gallery"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_to_b64_thumbnail(image: np.ndarray, max_size: int = 200) -> str:
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def main():
    print("=" * 60)
    print("  Face Recognition - Gallery Seed Script")
    print("=" * 60)

    # ── 1. Connect to MongoDB ─────────────────────────────────────────────────
    print("\n[1/4] Connecting to MongoDB Atlas...")
    from web.backend.services.mongo_service import connect, get_db, is_connected
    success = connect()   # must call connect() explicitly first!
    db = get_db()
    if not success or not is_connected():
        print("ERROR: Cannot connect to MongoDB Atlas!")
        print("  Make sure your .env file has MONGO_URI set correctly.")
        mongo_uri = os.environ.get("MONGO_URI", "NOT SET")
        print(f"  MONGO_URI = {mongo_uri[:50]}...")
        sys.exit(1)
    print("OK: MongoDB connected!")

    # ── 2. Load the face recognition pipeline ────────────────────────────────
    print("\n[2/4] Loading InsightFace pipeline (takes ~30 seconds)...")
    from web.backend.services import pipeline_service
    pipeline_service.load_pipeline()
    pipe = pipeline_service.get_pipeline()
    if pipe is None:
        print("ERROR: Pipeline failed to load!")
        sys.exit(1)
    print("OK: Pipeline loaded!")

    # ── 3. Import embedding service ───────────────────────────────────────────
    from web.backend.services.embedding_service import store_embedding, list_identities

    # ── 4. Process each identity folder ──────────────────────────────────────
    print("\n[3/4] Processing gallery images...\n")

    identity_dirs = sorted([d for d in GALLERY_DIR.iterdir() if d.is_dir()])
    if not identity_dirs:
        print("ERROR: No identity folders found in data/gallery/")
        sys.exit(1)

    total_saved = 0
    total_errors = 0

    for id_dir in identity_dirs:
        identity = id_dir.name
        images = sorted([
            f for f in id_dir.iterdir()
            if f.suffix.lower() in ALLOWED_EXTENSIONS
        ])

        print(f"  IDENTITY: {identity.upper()} -- {len(images)} images")
        saved = 0
        errors = 0

        for img_path in images:
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"    [SKIP] Could not read: {img_path.name}")
                    errors += 1
                    continue

                # Gallery images are pre-cropped face chips (already aligned).
                # Skip SCRFD detection — feed directly into ArcFace backbone.
                emb = pipe.arc_ext.extract_from_embedding_model(image)

                # Check it's not a zero vector (happens for larger full-photo images)
                import numpy as _np
                if emb is None or _np.linalg.norm(emb) < 1e-6:
                    # Fallback: try full detect → align → embed pipeline
                    detection = pipe.detector.detect_largest(image)
                    if detection is not None:
                        aligned = pipe.aligner.align_from_detection(image, detection)
                        if aligned is not None:
                            emb = pipe.arc_ext.extract(aligned)

                # Final check
                if emb is None or _np.linalg.norm(emb) < 1e-6:
                    print(f"    [SKIP] Could not extract embedding: {img_path.name}")
                    errors += 1
                    continue

                thumb_b64 = image_to_b64_thumbnail(image)

                ok = store_embedding(
                    identity=identity,
                    embedding=emb,
                    image_b64=thumb_b64,
                    filename=img_path.name,
                )
                if ok:
                    saved += 1
                    print(f"    [OK] {img_path.name}")
                else:
                    print(f"    [FAIL] Could not store: {img_path.name}")
                    errors += 1

            except Exception as e:
                print(f"    [ERROR] {img_path.name}: {e}")
                errors += 1

        print(f"    --> Saved: {saved}, Errors: {errors}\n")
        total_saved += saved
        total_errors += errors

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print("[4/4] Done!\n")
    print("=" * 60)
    print(f"  Total embeddings stored : {total_saved}")
    print(f"  Total errors            : {total_errors}")
    print()

    identities = list_identities()
    print(f"  Identities now in MongoDB ({len(identities)}):")
    for ident in identities:
        print(f"    - {ident['name']} -- {ident.get('image_count', '?')} embeddings")

    print()
    print("  GALLERY IS NOW PRELOADED!")
    print("  Scientists visiting your URL will see all 3 identities.")
    print("=" * 60)


if __name__ == "__main__":
    main()
