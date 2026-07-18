"""
embedding_cache.py
------------------
Hash-based feature vector cache for gallery images.

WHY: Every retrain re-extracts ALL features from scratch - even for images that
     have not changed. 45 gallery images x 350 ms each = ~16 s just for feature
     extraction. This cache reduces that to ~1-2 s (SVM fit only) for unchanged
     images after the first run.

HOW:
  - Each image gets an MD5 hash of its file contents on load.
  - The hash + feature vector are stored in a .cache/ subdirectory beside the image.
  - On the next retrain, if the hash still matches, the .npy is loaded in ~1 ms.
  - If the image was changed or is new, the cache is rebuilt automatically.
  - Cache is per-modality-set so changing modalities forces a full re-extract.

Face Recognition on Skewed UAV Images
"""

import hashlib
import numpy as np
from pathlib import Path
from typing import Optional


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _modalities_tag(modalities: list) -> str:
    return "_".join(sorted(modalities))


def get_cached_feature(img_path: Path, modalities: list) -> Optional[np.ndarray]:
    """Return cached feature vector if valid, else None."""
    cache_dir = img_path.parent / ".cache"
    tag       = _modalities_tag(modalities)
    npy_path  = cache_dir / f"{img_path.name}_{tag}.npy"
    hash_path = cache_dir / f"{img_path.name}_{tag}.hash"
    if not npy_path.exists() or not hash_path.exists():
        return None
    try:
        if _file_md5(img_path) != hash_path.read_text(encoding="utf-8").strip():
            return None   # stale
        return np.load(str(npy_path)).astype(np.float32)
    except Exception:
        return None


def save_cached_feature(img_path: Path, modalities: list, feature: np.ndarray) -> None:
    """Write feature vector + hash to cache. Silently skips on any error."""
    try:
        cache_dir = img_path.parent / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        tag = _modalities_tag(modalities)
        np.save(str(cache_dir / f"{img_path.name}_{tag}.npy"), feature.astype(np.float32))
        (cache_dir / f"{img_path.name}_{tag}.hash").write_text(
            _file_md5(img_path), encoding="utf-8"
        )
    except Exception:
        pass   # cache write failure is non-critical


def clear_cache(gallery_dir: Path, modalities: list = None) -> int:
    """Delete all cached files under gallery_dir. Returns count deleted."""
    deleted = 0
    tag = _modalities_tag(modalities) if modalities else None
    for cache_dir in gallery_dir.rglob(".cache"):
        for f in cache_dir.iterdir():
            if tag is None or tag in f.name:
                try:
                    f.unlink()
                    deleted += 1
                except Exception:
                    pass
    return deleted


def cache_stats(gallery_dir: Path, modalities: list) -> dict:
    """Return hit/miss statistics dict for logging."""
    tag = _modalities_tag(modalities)
    total = cached = stale = 0
    for img_path in gallery_dir.rglob("*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        if ".cache" in img_path.parts:
            continue
        total += 1
        h_file = img_path.parent / ".cache" / f"{img_path.name}_{tag}.hash"
        n_file = img_path.parent / ".cache" / f"{img_path.name}_{tag}.npy"
        if n_file.exists() and h_file.exists():
            try:
                cur = _file_md5(img_path)
                stored = h_file.read_text().strip()
                if cur == stored:
                    cached += 1
                else:
                    stale += 1
            except Exception:
                stale += 1
    return {
        "total":    total,
        "cached":   cached,
        "stale":    stale,
        "hit_rate": f"{cached / total * 100:.0f}%" if total else "N/A",
    }
