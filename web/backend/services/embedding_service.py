"""
embedding_service.py
--------------------
Stores and retrieves ArcFace face embeddings in MongoDB Atlas.
Used for live recognition (webcam, identify, batch) without needing SVM/PCA.

Recognition method: cosine similarity
  - Each enrolled face has a 512-D ArcFace embedding stored in MongoDB.
  - At inference time, the query embedding is compared against all stored
    embeddings. The closest match above the threshold is returned.
  - No retraining needed when adding new faces — just upload and it works.

Falls back to SVM pipeline if MongoDB is not connected.
"""

import base64
import numpy as np
from datetime import datetime
from typing import Optional

from web.backend.services.mongo_service import get_db, is_connected


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


# ── Write ──────────────────────────────────────────────────────────────────────

def store_embedding(
    identity:  str,
    embedding: np.ndarray,
    image_b64: Optional[str] = None,
    filename:  str = "",
) -> bool:
    """
    Save a face embedding to MongoDB.

    Parameters
    ----------
    identity  : person name / label
    embedding : 512-D ArcFace feature vector (numpy array)
    image_b64 : base64-encoded JPEG thumbnail (optional, for gallery display)
    filename  : original filename for reference

    Returns True if stored successfully, False otherwise.
    """
    db = get_db()
    if db is None:
        return False
    try:
        doc = {
            "identity":   identity,
            "embedding":  embedding.tolist(),   # store as plain list
            "image_b64":  image_b64 or "",
            "filename":   filename,
            "created_at": datetime.utcnow().isoformat(),
        }
        db["gallery_embeddings"].insert_one(doc)
        return True
    except Exception as e:
        print(f"[EmbeddingService] store_embedding error: {e}")
        return False


def delete_identity(name: str) -> int:
    """Delete all embeddings for a given identity. Returns count deleted."""
    db = get_db()
    if db is None:
        return 0
    try:
        result = db["gallery_embeddings"].delete_many({"identity": name})
        return result.deleted_count
    except Exception as e:
        print(f"[EmbeddingService] delete_identity error: {e}")
        return 0


# ── Read ───────────────────────────────────────────────────────────────────────

def list_identities() -> list[dict]:
    """
    Return list of {name, image_count} for all enrolled identities.
    """
    db = get_db()
    if db is None:
        return []
    try:
        pipeline = [
            {"$group": {
                "_id":         "$identity",
                "image_count": {"$sum": 1},
                "latest":      {"$max": "$created_at"},
            }},
            {"$sort": {"_id": 1}},
        ]
        results = list(db["gallery_embeddings"].aggregate(pipeline))
        return [
            {"name": r["_id"], "image_count": r["image_count"], "latest": r["latest"]}
            for r in results
        ]
    except Exception as e:
        print(f"[EmbeddingService] list_identities error: {e}")
        return []


def get_identity_images(name: str, max_images: int = 8) -> list[dict]:
    """Return base64 thumbnail images for a given identity."""
    db = get_db()
    if db is None:
        return []
    try:
        docs = list(db["gallery_embeddings"].find(
            {"identity": name, "image_b64": {"$ne": ""}},
            {"image_b64": 1, "filename": 1, "_id": 0}
        ).limit(max_images))
        return [{"filename": d["filename"], "data": d["image_b64"]} for d in docs]
    except Exception as e:
        print(f"[EmbeddingService] get_identity_images error: {e}")
        return []


def get_all_embeddings() -> list[dict]:
    """Load ALL stored embeddings from MongoDB (used for similarity search)."""
    db = get_db()
    if db is None:
        return []
    try:
        docs = list(db["gallery_embeddings"].find(
            {}, {"identity": 1, "embedding": 1, "_id": 0}
        ))
        return docs
    except Exception as e:
        print(f"[EmbeddingService] get_all_embeddings error: {e}")
        return []


# ── Inference ──────────────────────────────────────────────────────────────────

def find_best_match(
    query_embedding: np.ndarray,
    threshold: float = 0.35,
) -> dict:
    """
    Find the best matching identity for a query face embedding.

    Loads all stored embeddings and performs cosine similarity search.
    Returns the best match if similarity >= threshold, else UNKNOWN.

    Parameters
    ----------
    query_embedding : 512-D ArcFace embedding from the query face
    threshold       : minimum cosine similarity to count as "known"

    Returns
    -------
    {
        "identity":   str,
        "confidence": float,
        "is_known":   bool,
    }
    """
    stored = get_all_embeddings()
    if not stored:
        return {"identity": "UNKNOWN", "confidence": 0.0, "is_known": False}

    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

    best_score    = -1.0
    best_identity = "UNKNOWN"

    for doc in stored:
        emb = np.array(doc["embedding"], dtype=np.float32)
        emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
        score = float(np.dot(query_norm, emb_norm))
        if score > best_score:
            best_score    = score
            best_identity = doc["identity"]

    is_known = best_score >= threshold
    return {
        "identity":   best_identity if is_known else "UNKNOWN",
        "confidence": round(best_score, 4),
        "is_known":   is_known,
    }


def total_embeddings() -> int:
    """Return total number of stored face embeddings."""
    db = get_db()
    if db is None:
        return 0
    try:
        return db["gallery_embeddings"].count_documents({})
    except Exception:
        return 0
