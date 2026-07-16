"""
matcher.py
----------
WHAT : FAISS-based cosine similarity matcher for face identity lookup.

WHY FAISS INSTEAD OF SVM:
  - SVM is a closed-set classifier — it always assigns to a gallery identity
    even for unknown people (false positives). FAISS with a threshold naturally
    handles open-set: if cosine_sim < threshold, return "UNKNOWN".
  - SVM scales poorly beyond ~200 identities (slow predict_proba).
    FAISS searches 100K embeddings in under 10ms on CPU.
  - ArcFace embeddings are designed for cosine similarity — FAISS uses the
    same matching method ArcFace was trained with (maximise intra-class cosine
    similarity, minimise inter-class).

HOW IT INTEGRATES:
  - At retrain time: gallery ArcFace embeddings → FAISSMatcher.add_gallery()
  - At inference time: query ArcFace embedding → FAISSMatcher.search()
  - The SVM pipeline still runs in parallel for the CMC/ranked output;
    FAISS is used as the primary accept/reject gate.

Face Recognition on Skewed UAV Images
"""

import numpy as np
from pathlib import Path
import joblib
from typing import Optional


class FAISSMatcher:
    """
    Cosine similarity identity matcher backed by a FAISS flat index.

    Uses IndexFlatIP (inner product on L2-normalised vectors = cosine similarity).
    No GPU required — faiss-cpu runs this in <1ms for galleries up to 50K entries.

    Parameters
    ----------
    threshold : float
        Minimum cosine similarity to accept a match. Vectors with best similarity
        below this value are returned as "UNKNOWN".
        Recommended: 0.60 for general use, 0.65 for high-security watchlist.
    dim : int
        ArcFace embedding dimensionality. buffalo_l uses 512-D.
    """

    def __init__(self, threshold: float = 0.60, dim: int = 512):
        self.threshold = threshold
        self.dim = dim
        self._index = None      # faiss.IndexFlatIP built at add_gallery time
        self._labels: list[str] = []   # identity label for each gallery vector
        self._built = False

    # ── Building the gallery index ────────────────────────────────────────────

    def add_gallery(
        self,
        embeddings: np.ndarray,
        labels: list[str],
    ) -> "FAISSMatcher":
        """
        Build FAISS index from gallery ArcFace embeddings.

        Parameters
        ----------
        embeddings : np.ndarray of shape (n, 512)
            Raw ArcFace embeddings for all gallery images.
        labels : list[str]
            Identity label for each embedding (same order).

        Returns
        -------
        self — for method chaining
        """
        try:
            import faiss
        except ImportError:
            print("[FAISSMatcher] faiss-cpu not installed. Run: pip install faiss-cpu")
            return self

        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]

        # L2-normalise so inner-product = cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)   # avoid /0
        normed = (embeddings / norms).astype(np.float32)

        dim = normed.shape[1]
        self._index = faiss.IndexFlatIP(dim)   # Inner Product index (cosine on L2-normed vecs)
        self._index.add(normed)
        self._labels = list(labels)
        self._built = True

        print(f"[FAISSMatcher] Index built: {len(labels)} gallery vectors, dim={dim}, "
              f"threshold={self.threshold}")
        return self

    def add_identity(self, identity: str, embeddings: np.ndarray) -> None:
        """
        Add or update a single identity's embeddings to the index.
        Rebuilds the full index — call sparingly (only at re-enroll time).
        """
        # Remove existing entries for this identity
        keep = [(lbl, i) for i, lbl in enumerate(self._labels) if lbl != identity]

        existing_labels = [k[0] for k in keep]
        existing_idx    = [k[1] for k in keep]

        if self._index is not None and existing_idx:
            try:
                import faiss
                # Reconstruct existing vectors
                old_vecs = np.zeros((self._index.ntotal, self.dim), dtype=np.float32)
                for i in range(self._index.ntotal):
                    self._index.reconstruct(i, old_vecs[i])
                kept_vecs = old_vecs[existing_idx]
            except Exception:
                kept_vecs = np.zeros((0, self.dim), dtype=np.float32)
        else:
            kept_vecs = np.zeros((0, self.dim), dtype=np.float32)

        # Normalise new embeddings
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        new_vecs = (embeddings / norms).astype(np.float32)

        all_vecs   = np.vstack([kept_vecs, new_vecs]) if len(kept_vecs) else new_vecs
        all_labels = existing_labels + [identity] * len(new_vecs)

        self.add_gallery(all_vecs, all_labels)

    # ── Inference ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Find the top-k most similar gallery identities for a query embedding.

        Parameters
        ----------
        query : np.ndarray of shape (512,) or (1, 512)
            ArcFace embedding of the query face (raw, unnormalized).
        top_k : int
            Number of candidates to return.

        Returns
        -------
        list of dicts: [{"identity": str, "similarity": float, "is_known": bool}, ...]
        sorted by similarity descending.
        Returns [{"identity": "UNKNOWN", "similarity": 0.0, "is_known": False}]
        if index is empty or best similarity < threshold.
        """
        if not self._built or self._index is None or self._index.ntotal == 0:
            return [{"identity": "UNKNOWN", "similarity": 0.0, "is_known": False}]

        try:
            import faiss
        except ImportError:
            return [{"identity": "UNKNOWN", "similarity": 0.0, "is_known": False}]

        q = np.array(query, dtype=np.float32).ravel()
        norm = np.linalg.norm(q)
        if norm < 1e-8:
            return [{"identity": "UNKNOWN", "similarity": 0.0, "is_known": False}]
        q = (q / norm)[np.newaxis, :]   # shape (1, 512), L2-normed

        k = min(top_k, self._index.ntotal)
        similarities, indices = self._index.search(q, k)   # shapes (1, k)

        results = []
        seen_identities = set()

        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx >= len(self._labels):
                continue
            identity = self._labels[idx]
            # De-duplicate: keep only best score per identity
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            is_known = float(sim) >= self.threshold
            results.append({
                "identity":   identity if is_known else "UNKNOWN",
                "similarity": round(float(sim), 4),
                "is_known":   is_known,
            })

        if not results or results[0]["similarity"] < self.threshold:
            return [{"identity": "UNKNOWN", "similarity": results[0]["similarity"] if results else 0.0,
                     "is_known": False}]

        return results

    def best_match(self, query: np.ndarray) -> tuple[str, float]:
        """
        Return (identity, similarity) for the single best match.
        Returns ("UNKNOWN", similarity) if below threshold.
        """
        results = self.search(query, top_k=1)
        r = results[0]
        return r["identity"], r["similarity"]

    def search_ranked(self, query: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Return top-k ranked gallery matches ALWAYS showing real identity names.

        Unlike search() which replaces below-threshold identities with "UNKNOWN"
        (correct for open-set primary identification), this method always shows
        the actual identity name alongside its similarity score.

        WHY: The ranked candidates UI list is for user inspection — users want to
        see that Siddhant scored 0.69 and Aditi scored 0.51, not just "UNKNOWN".
        The is_known flag still indicates whether the score clears the threshold.

        Returns
        -------
        list of dicts: [{"identity": str, "similarity": float, "is_known": bool}, ...]
        sorted by similarity descending (real names, never masked to UNKNOWN).
        """
        if not self._built or self._index is None or self._index.ntotal == 0:
            return []

        try:
            import faiss
        except ImportError:
            return []

        q = np.array(query, dtype=np.float32).ravel()
        norm = np.linalg.norm(q)
        if norm < 1e-8:
            return []
        q = (q / norm)[np.newaxis, :]  # shape (1, 512), L2-normed

        k = min(top_k * 3, self._index.ntotal)  # fetch extra to de-dup by identity
        similarities, indices = self._index.search(q, k)

        results = []
        seen_identities = set()

        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0 or idx >= len(self._labels):
                continue
            identity = self._labels[idx]
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            results.append({
                "identity":   identity,          # always the real name, never "UNKNOWN"
                "similarity": round(float(sim), 4),
                "is_known":   float(sim) >= self.threshold,
            })
            if len(results) >= top_k:
                break

        return results

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_built(self) -> bool:
        return self._built and self._index is not None

    @property
    def n_gallery(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def n_identities(self) -> int:
        return len(set(self._labels))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the FAISS index + labels to disk."""
        try:
            import faiss
        except ImportError:
            print("[FAISSMatcher] Cannot save — faiss not installed.")
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save index
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        # Save labels + config
        joblib.dump({
            "labels":    self._labels,
            "threshold": self.threshold,
            "dim":       self.dim,
            "built":     self._built,
        }, path.with_suffix(".pkl"))
        print(f"[FAISSMatcher] Saved to {path.with_suffix('.faiss')} + .pkl")

    @classmethod
    def load(cls, path: str | Path) -> "FAISSMatcher":
        """Load a previously saved FAISS matcher from disk."""
        try:
            import faiss
        except ImportError:
            print("[FAISSMatcher] Cannot load — faiss not installed.")
            return cls()

        path = Path(path)
        faiss_path = path.with_suffix(".faiss")
        pkl_path   = path.with_suffix(".pkl")

        if not faiss_path.exists() or not pkl_path.exists():
            print(f"[FAISSMatcher] Index files not found at {path}. Will build at next retrain.")
            return cls()

        data  = joblib.load(pkl_path)
        obj   = cls(threshold=data["threshold"], dim=data["dim"])
        obj._labels = data["labels"]
        obj._built  = data["built"]
        obj._index  = faiss.read_index(str(faiss_path))
        print(f"[FAISSMatcher] Loaded: {obj.n_gallery} vectors, "
              f"{obj.n_identities} identities, threshold={obj.threshold}")
        return obj
