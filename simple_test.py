"""
Simple test script for components that don't require ML models.
"""
import os
import numpy as np
from encryption import EmbeddingEncryptor
from chromadb_storage import ChromaDBStorage
import tempfile

def test_encryption():
    """Test AES encryption functionality."""
    print("Testing encryption...")
    encryptor = EmbeddingEncryptor()

    # Create test embedding
    test_embedding = np.random.randn(512).astype(np.float32)

    # Test encryption
    encrypted_data, iv = encryptor.encrypt_embedding(test_embedding)
    print("   ✅ Encryption successful")

    # Test decryption
    decrypted_embedding = encryptor.decrypt_embedding(encrypted_data, iv)
    if np.allclose(test_embedding, decrypted_embedding, rtol=1e-5):
        print("   ✅ Decryption successful")
    else:
        print("   ❌ Decryption failed")

def test_chromadb_storage():
    """Test ChromaDB storage functionality."""
    print("Testing ChromaDB storage...")

    # Use temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = ChromaDBStorage(persist_directory=temp_dir)

        # Create test data
        test_embedding = np.random.randn(100).astype(np.uint8).tobytes()
        user_id = "test_user"

        # Test adding embedding
        embedding_id = storage.add_embedding(
            user_id=user_id,
            encrypted_embedding=test_embedding,
            metadata={"test": True}
        )
        print("   ✅ Added embedding")

        # Test getting by user
        user_embeddings = storage.get_embeddings_by_user(user_id)
        print(f"   ✅ Retrieved {len(user_embeddings)} embeddings")

        # Test collection stats
        stats = storage.get_collection_stats()
        print(f"   ✅ Collection has {stats['total_embeddings']} embeddings")

if __name__ == "__main__":
    print("Simple Component Tests")
    print("=" * 30)

    test_encryption()
    print()

    test_chromadb_storage()
    print()

    print("✅ Simple tests completed!")
