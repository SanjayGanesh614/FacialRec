#!/usr/bin/env python3
"""
Test embedding extraction and matching directly.
"""
import sys
import os
import numpy as np
import cv2

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core.face_embedding import FaceEmbedder
from src.core.encryption import EmbeddingEncryptor
from src.core.chromadb_storage import ChromaDBStorage

def create_simple_face_patch():
    """Create a simple face-like patch for embedding testing."""
    # Create a 160x160 patch (standard face size)
    patch = np.random.randint(50, 200, (160, 160, 3), dtype=np.uint8)
    
    # Add some structure to make it more face-like
    center_x, center_y = 80, 80
    
    # Add some circular patterns (eye-like)
    cv2.circle(patch, (center_x-30, center_y-20), 15, (100, 100, 100), -1)
    cv2.circle(patch, (center_x+30, center_y-20), 15, (100, 100, 100), -1)
    
    # Add some lines (nose/mouth-like)
    cv2.line(patch, (center_x, center_y), (center_x, center_y+20), (120, 120, 120), 2)
    cv2.ellipse(patch, (center_x, center_y+30), (20, 10), 0, 0, 180, (80, 80, 80), 2)
    
    return patch

def test_embedding_extraction():
    """Test embedding extraction."""
    print("🧪 Testing Embedding Extraction")
    print("=" * 40)
    
    # Initialize embedder
    print("1. Initializing FaceEmbedder...")
    embedder = FaceEmbedder(device='cpu')
    
    # Create test patches
    print("2. Creating test face patches...")
    patch1 = create_simple_face_patch()
    patch2 = create_simple_face_patch()  # Similar but different
    
    # Extract embeddings
    print("3. Extracting embeddings...")
    embedding1 = embedder.extract_embedding(patch1)
    embedding2 = embedder.extract_embedding(patch2)
    
    if embedding1 is None or embedding2 is None:
        print("❌ Failed to extract embeddings")
        return False
    
    print(f"   Embedding 1 shape: {embedding1.shape}")
    print(f"   Embedding 2 shape: {embedding2.shape}")
    print(f"   Embedding 1 norm: {np.linalg.norm(embedding1):.4f}")
    print(f"   Embedding 2 norm: {np.linalg.norm(embedding2):.4f}")
    
    # Test similarity
    print("4. Testing similarity calculation...")
    similarity = embedder.compute_similarity(embedding1, embedding2)
    print(f"   Similarity between embeddings: {similarity:.4f}")
    
    # Test self-similarity
    self_similarity = embedder.compute_similarity(embedding1, embedding1)
    print(f"   Self-similarity: {self_similarity:.4f}")
    
    return True

def test_encryption_decryption():
    """Test encryption and decryption."""
    print("\n🧪 Testing Encryption/Decryption")
    print("=" * 40)
    
    # Initialize components
    print("1. Initializing components...")
    embedder = FaceEmbedder(device='cpu')
    encryptor = EmbeddingEncryptor()
    
    # Create test embedding
    print("2. Creating test embedding...")
    patch = create_simple_face_patch()
    embedding = embedder.extract_embedding(patch)
    
    if embedding is None:
        print("❌ Failed to create embedding")
        return False
    
    print(f"   Original embedding shape: {embedding.shape}")
    print(f"   Original embedding norm: {np.linalg.norm(embedding):.4f}")
    
    # Test encryption
    print("3. Testing encryption...")
    encrypted_data, iv = encryptor.encrypt_embedding(embedding)
    print(f"   Encrypted data length: {len(encrypted_data)}")
    print(f"   IV length: {len(iv)}")
    
    # Test decryption
    print("4. Testing decryption...")
    decrypted_embedding = encryptor.decrypt_embedding(encrypted_data, iv)
    
    if decrypted_embedding is None:
        print("❌ Failed to decrypt embedding")
        return False
    
    print(f"   Decrypted embedding shape: {decrypted_embedding.shape}")
    print(f"   Decrypted embedding norm: {np.linalg.norm(decrypted_embedding):.4f}")
    
    # Test if decryption is correct
    similarity = embedder.compute_similarity(embedding, decrypted_embedding)
    print(f"   Similarity original vs decrypted: {similarity:.4f}")
    
    if similarity > 0.99:
        print("   ✅ Encryption/decryption working correctly")
        return True
    else:
        print("   ❌ Encryption/decryption has issues")
        return False

def test_storage_and_search():
    """Test storage and similarity search."""
    print("\n🧪 Testing Storage and Search")
    print("=" * 40)
    
    # Initialize components
    print("1. Initializing components...")
    embedder = FaceEmbedder(device='cpu')
    encryptor = EmbeddingEncryptor()
    storage = ChromaDBStorage()
    
    # Reset storage for clean test
    print("2. Resetting storage...")
    storage.reset_collection()
    
    # Create test embeddings
    print("3. Creating test embeddings...")
    patch1 = create_simple_face_patch()
    patch2 = create_simple_face_patch()
    
    embedding1 = embedder.extract_embedding(patch1)
    embedding2 = embedder.extract_embedding(patch2)
    
    if embedding1 is None or embedding2 is None:
        print("❌ Failed to create embeddings")
        return False
    
    # Encrypt and store embeddings
    print("4. Encrypting and storing embeddings...")
    encrypted1, iv1 = encryptor.encrypt_embedding(embedding1)
    encrypted2, iv2 = encryptor.encrypt_embedding(embedding2)
    
    # Combine IV and encrypted data as done in the main code
    combined1 = iv1 + encrypted1
    combined2 = iv2 + encrypted2
    
    # Store in database
    id1 = storage.add_embedding("user1", combined1, {"name": "Test User 1"})
    id2 = storage.add_embedding("user2", combined2, {"name": "Test User 2"})
    
    print(f"   Stored embedding 1 with ID: {id1}")
    print(f"   Stored embedding 2 with ID: {id2}")
    
    # Test search
    print("5. Testing similarity search...")
    
    # Search for similar to embedding1
    query_encrypted, query_iv = encryptor.encrypt_embedding(embedding1)
    query_combined = query_iv + query_encrypted
    
    results = storage.search_similar(query_combined, n_results=5, threshold=0.0)
    
    print(f"   Found {len(results)} results")
    for i, result in enumerate(results):
        print(f"   Result {i+1}: {result['user_id']} (similarity: {result['similarity']:.4f})")
    
    # Check if we found the correct match
    if len(results) > 0 and results[0]['user_id'] == 'user1' and results[0]['similarity'] > 0.8:
        print("   ✅ Storage and search working correctly")
        return True
    else:
        print("   ❌ Storage and search have issues")
        return False

def main():
    """Run all embedding and matching tests."""
    print("🔬 Embedding and Matching Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    try:
        if test_embedding_extraction():
            tests_passed += 1
        
        if test_encryption_decryption():
            tests_passed += 1
        
        if test_storage_and_search():
            tests_passed += 1
        
        print("\n" + "=" * 50)
        print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
        
        if tests_passed == total_tests:
            print("🎉 All embedding and matching tests PASSED!")
            print("The core functionality is working correctly!")
            return 0
        else:
            print("❌ Some tests failed. Core functionality needs fixing.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())