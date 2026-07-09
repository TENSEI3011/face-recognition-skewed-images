"""
mongo_service.py
----------------
MongoDB Atlas connection singleton.

Provides get_db() used by all services.
Gracefully falls back (returns None) if MONGO_URI is not set,
so local development works without MongoDB.
"""

import os
import threading
from typing import Optional

_client = None
_db     = None
_lock   = threading.Lock()

MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME   = os.getenv("MONGO_DB_NAME", "facerecog_db")


def connect() -> bool:
    """
    Connect to MongoDB Atlas. Called once at server startup.
    Returns True if connection succeeded, False if MongoDB is not configured.
    """
    global _client, _db

    if not MONGO_URI:
        print("[MongoDB] MONGO_URI not set — running in local-file mode.")
        return False

    with _lock:
        if _client is not None:
            return True
        try:
            from pymongo import MongoClient
            from pymongo.server_api import ServerApi
            _client = MongoClient(MONGO_URI, server_api=ServerApi("1"), serverSelectionTimeoutMS=5000)
            # Ping to verify connection
            _client.admin.command("ping")
            _db = _client[DB_NAME]
            # Create indexes for fast queries
            _db["gallery_embeddings"].create_index("identity")
            _db["audit_log"].create_index([("timestamp", -1)])
            _db["watchlist"].create_index("name", unique=True)
            print(f"[MongoDB] OK Connected to Atlas -- database: {DB_NAME}")
            return True
        except Exception as e:
            print(f"[MongoDB] FAILED Connection failed: {e}")
            _client = None
            _db     = None
            return False


def get_db():
    """Return the database handle, or None if not connected."""
    return _db


def is_connected() -> bool:
    return _db is not None
