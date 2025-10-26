#!/usr/bin/env python3
"""
Reset the ChromaDB database to fix dimension issues.
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core.chromadb_storage import ChromaDBStorage

def main():
    """Reset the database."""
    print("🗄️ Resetting ChromaDB database...")
    
    try:
        storage = ChromaDBStorage()
        storage.reset_collection()
        print("✅ Database reset successfully!")
        
        # Check stats
        stats = storage.get_collection_stats()
        print(f"📊 New collection stats: {stats}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error resetting database: {e}")
        return 1

if __name__ == "__main__":
    exit(main())