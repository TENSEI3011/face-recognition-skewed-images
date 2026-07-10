"""
cleanup_mongo.py - Remove duplicate embeddings from MongoDB Atlas.

Run this to deduplicate the gallery_embeddings collection.
Each identity+filename should only appear once.

Run with:
    python cleanup_mongo.py
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from web.backend.services.mongo_service import connect, get_db, is_connected

def main():
    print("=" * 55)
    print("  MongoDB Gallery Deduplication")
    print("=" * 55)

    print("\nConnecting to MongoDB Atlas...")
    connect()
    if not is_connected():
        print("ERROR: Cannot connect to MongoDB!")
        sys.exit(1)
    print("OK: Connected!\n")

    db = get_db()
    col = db["gallery_embeddings"]

    # Count before
    total_before = col.count_documents({})
    print(f"Total embeddings before cleanup: {total_before}")

    # Group by identity+filename, keep only the first (oldest) per combo
    pipeline = [
        {"$group": {
            "_id": {"identity": "$identity", "filename": "$filename"},
            "ids":  {"$push": "$_id"},
            "count": {"$sum": 1}
        }},
        {"$match": {"count": {"$gt": 1}}}  # Only duplicates
    ]

    duplicates = list(col.aggregate(pipeline))
    print(f"Duplicate groups found: {len(duplicates)}")

    removed = 0
    for dup in duplicates:
        # Keep the first ID, delete the rest
        ids_to_delete = dup["ids"][1:]   # skip index 0 (keep it)
        result = col.delete_many({"_id": {"$in": ids_to_delete}})
        removed += result.deleted_count

    total_after = col.count_documents({})
    print(f"\nRemoved {removed} duplicate embeddings.")
    print(f"Total embeddings after cleanup: {total_after}")

    # Show final state
    from web.backend.services.embedding_service import list_identities
    identities = list_identities()
    print(f"\nFinal gallery ({len(identities)} identities):")
    for ident in identities:
        print(f"  - {ident['name']}: {ident['image_count']} embeddings")

    print("\nDone! Gallery is clean.")
    print("=" * 55)

if __name__ == "__main__":
    main()
