"""
Test script for the facial recognition system.
"""
import os
import sys
import numpy as np
from pathlib import Path
import tempfile
import shutil
from matching_module import FaceMatcher
import config


def create_test_image(size=(160, 160), color=(128, 128, 128)):
    """Create a simple test image."""
    import cv2
    image = np.full((*size, 3), color, dtype=np.uint8)
    return image


def test_face_detection():
    """Test face detection functionality."""
    print("Testing face detection...")
    
    from face_detection import FaceDetector
    
    detector = FaceDetector(device='cpu')
    
    # Create a test image
    test_image = create_test_image()
    
    # Test face detection
    faces = detector.detect_faces(test_image)
    print(f"   Detected {len(faces)} faces in test image")
    
    # Test face extraction
    if faces:
        face = detector.extract_face(test_image, faces[0])
        if face is not None:
            print("   ✅ Face extraction successful")
        else:
            print("   ❌ Face extraction failed")
    else:
        print("   ⚠️  No faces detected (expected for synthetic image)")
    
    print("   ✅ Face detection test completed")


def test_face_embedding():
    """Test face embedding extraction."""
    print("Testing face embedding...")
    
    from face_embedding import FaceEmbedder
    
    embedder = FaceEmbedder(device='cpu')
    
    # Create a test face image
    test_face = create_test_image()
    
    # Test embedding extraction
    embedding = embedder.extract_embedding(test_face)
    
    if embedding is not None:
        print(f"   ✅ Embedding extracted successfully")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Embedding norm: {np.linalg.norm(embedding):.4f}")
    else:
        print("   ❌ Embedding extraction failed")
    
    print("   ✅ Face embedding test completed")


def test_encryption():
    """Test AES encryption functionality."""
    print("Testing encryption...")
    
    from encryption import EmbeddingEncryptor
    
    encryptor = EmbeddingEncryptor()
    
    # Create test embedding
    test_embedding = np.random.randn(512).astype(np.float32)
    
    # Test encryption
    encrypted_data, iv = encryptor.encrypt_embedding(test_embedding)
    print(f"   ✅ Encryption successful")
    print(f"   Encrypted data size: {len(encrypted_data)} bytes")
    print(f"   IV size: {len(iv)} bytes")
    
    # Test decryption
    decrypted_embedding = encryptor.decrypt_embedding(encrypted_data, iv)
    
    if decrypted_embedding is not None:
        # Check if decrypted embedding matches original
        if np.allclose(test_embedding, decrypted_embedding, rtol=1e-5):
            print("   ✅ Decryption successful and data integrity verified")
        else:
            print("   ❌ Decrypted data does not match original")
    else:
        print("   ❌ Decryption failed")
    
    # Test string encoding/decoding
    encrypted_string = encryptor.encrypt_embedding_to_string(test_embedding)
    decrypted_from_string = encryptor.decrypt_embedding_from_string(encrypted_string)
    
    if decrypted_from_string is not None and np.allclose(test_embedding, decrypted_from_string, rtol=1e-5):
        print("   ✅ String encoding/decoding successful")
    else:
        print("   ❌ String encoding/decoding failed")
    
    print("   ✅ Encryption test completed")


def test_chromadb_storage():
    """Test ChromaDB storage functionality."""
    print("Testing ChromaDB storage...")
    
    from chromadb_storage import ChromaDBStorage
    
    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = ChromaDBStorage(persist_directory=temp_dir)
        
        # Create test data
        test_embedding = np.random.randn(100).astype(np.uint8)  # Simulate encrypted data
        user_id = "test_user"
        metadata = {"test": True}
        
        # Test adding embedding
        embedding_id = storage.add_embedding(
            user_id=user_id,
            encrypted_embedding=test_embedding.tobytes(),
            metadata=metadata
        )
        print(f"   ✅ Added embedding with ID: {embedding_id}")
        
        # Test searching
        similar_embeddings = storage.search_similar(
            query_encrypted_embedding=test_embedding.tobytes(),
            n_results=5
        )
        print(f"   ✅ Found {len(similar_embeddings)} similar embeddings")
        
        # Test getting by ID
        retrieved_embedding = storage.get_embedding_by_id(embedding_id)
        if retrieved_embedding:
            print("   ✅ Retrieved embedding by ID")
        else:
            print("   ❌ Failed to retrieve embedding by ID")
        
        # Test getting by user
        user_embeddings = storage.get_embeddings_by_user(user_id)
        print(f"   ✅ Retrieved {len(user_embeddings)} embeddings for user")
        
        # Test collection stats
        stats = storage.get_collection_stats()
        print(f"   ✅ Collection stats: {stats['total_embeddings']} embeddings")
    
    print("   ✅ ChromaDB storage test completed")


def test_matching_module():
    """Test the complete matching module."""
    print("Testing matching module...")
    
    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Update config to use temp directory
        original_persist_dir = config.CHROMA_PERSIST_DIRECTORY
        config.CHROMA_PERSIST_DIRECTORY = temp_dir
        
        try:
            matcher = FaceMatcher(device='cpu')
            
            # Create test images
            test_image1 = create_test_image()
            test_image2 = create_test_image()
            
            # Save test images
            import cv2
            test_image_path1 = os.path.join(temp_dir, "test1.jpg")
            test_image_path2 = os.path.join(temp_dir, "test2.jpg")
            cv2.imwrite(test_image_path1, test_image1)
            cv2.imwrite(test_image_path2, test_image2)
            
            # Test enrollment
            enroll_result = matcher.enroll_user(
                user_id="test_user",
                image_path=test_image_path1,
                metadata={"test": True}
            )
            
            if enroll_result['success']:
                print("   ✅ User enrollment successful")
            else:
                print(f"   ❌ User enrollment failed: {enroll_result['error']}")
            
            # Test authentication
            auth_result = matcher.authenticate_user(
                image_path=test_image_path2,
                threshold=0.5
            )
            
            if auth_result['success']:
                print(f"   ✅ Authentication completed (authenticated: {auth_result['authenticated']})")
            else:
                print(f"   ❌ Authentication failed: {auth_result['error']}")
            
            # Test face comparison
            compare_result = matcher.compare_faces(test_image_path1, test_image_path2)
            
            if compare_result['success']:
                print(f"   ✅ Face comparison successful (similarity: {compare_result['similarity']:.4f})")
            else:
                print(f"   ❌ Face comparison failed: {compare_result['error']}")
            
            # Test system stats
            stats_result = matcher.get_system_stats()
            if stats_result['success']:
                print(f"   ✅ System stats: {stats_result['total_embeddings']} embeddings")
            else:
                print(f"   ❌ Failed to get system stats: {stats_result['error']}")
        
        finally:
            # Restore original config
            config.CHROMA_PERSIST_DIRECTORY = original_persist_dir
    
    print("   ✅ Matching module test completed")


def run_all_tests():
    """Run all tests."""
    print("Facial Recognition System - Test Suite")
    print("=" * 50)
    
    try:
        test_face_detection()
        print()
        
        test_face_embedding()
        print()
        
        test_encryption()
        print()
        
        test_chromadb_storage()
        print()
        
        test_matching_module()
        print()
        
        print("=" * 50)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the facial recognition system')
    parser.add_argument('--component', choices=['detection', 'embedding', 'encryption', 'storage', 'matching'],
                       help='Test specific component only')
    
    args = parser.parse_args()
    
    if args.component == 'detection':
        test_face_detection()
    elif args.component == 'embedding':
        test_face_embedding()
    elif args.component == 'encryption':
        test_encryption()
    elif args.component == 'storage':
        test_chromadb_storage()
    elif args.component == 'matching':
        test_matching_module()
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
