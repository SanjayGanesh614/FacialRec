#!/usr/bin/env python3
"""
Test script to verify the facial recognition system setup.
"""
import sys
import os
import traceback

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print(f"✅ Pillow (PIL)")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    try:
        import chromadb
        print(f"✅ ChromaDB: {chromadb.__version__}")
    except ImportError as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False
    
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher
        print(f"✅ Cryptography")
    except ImportError as e:
        print(f"❌ Cryptography import failed: {e}")
        return False
    
    try:
        from flask import Flask
        print(f"✅ Flask")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    return True

def test_facenet_models():
    """Test if FaceNet models can be loaded."""
    print("\nTesting FaceNet models...")
    
    try:
        from facenet_pytorch import MTCNN, InceptionResnetV1
        print("✅ facenet-pytorch imported successfully")
        
        # Test MTCNN
        print("Loading MTCNN...")
        mtcnn = MTCNN(device='cpu')
        print("✅ MTCNN loaded successfully")
        
        # Test InceptionResnetV1
        print("Loading InceptionResnetV1...")
        model = InceptionResnetV1(pretrained='vggface2').eval()
        print("✅ InceptionResnetV1 loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ FaceNet model loading failed: {e}")
        traceback.print_exc()
        return False

def test_custom_modules():
    """Test if our custom modules can be imported."""
    print("\nTesting custom modules...")
    
    try:
        from src.core import config
        print("✅ config module imported")
    except ImportError as e:
        print(f"❌ config import failed: {e}")
        return False
    
    try:
        from src.core.face_detection import FaceDetector
        print("✅ FaceDetector imported")
    except ImportError as e:
        print(f"❌ FaceDetector import failed: {e}")
        return False
    
    try:
        from src.core.face_embedding import FaceEmbedder
        print("✅ FaceEmbedder imported")
    except ImportError as e:
        print(f"❌ FaceEmbedder import failed: {e}")
        return False
    
    try:
        from src.core.encryption import EmbeddingEncryptor
        print("✅ EmbeddingEncryptor imported")
    except ImportError as e:
        print(f"❌ EmbeddingEncryptor import failed: {e}")
        return False
    
    try:
        from src.core.chromadb_storage import ChromaDBStorage
        print("✅ ChromaDBStorage imported")
    except ImportError as e:
        print(f"❌ ChromaDBStorage import failed: {e}")
        return False
    
    try:
        from src.core.matching_module import FaceMatcher
        print("✅ FaceMatcher imported")
    except ImportError as e:
        print(f"❌ FaceMatcher import failed: {e}")
        return False
    
    return True

def test_system_initialization():
    """Test if the system can be initialized."""
    print("\nTesting system initialization...")
    
    try:
        from src.core.matching_module import FaceMatcher
        
        print("Initializing FaceMatcher...")
        matcher = FaceMatcher(device='cpu')  # Use CPU for testing
        print("✅ FaceMatcher initialized successfully")
        
        # Test system stats
        stats = matcher.get_system_stats()
        print(f"✅ System stats: {stats}")
        
        return True
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Facial Recognition System Setup Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test FaceNet models
    if not test_facenet_models():
        all_passed = False
    
    # Test custom modules
    if not test_custom_modules():
        all_passed = False
    
    # Test system initialization
    if not test_system_initialization():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nTo start the web server, run:")
        print("python web_server.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTo install missing dependencies, run:")
        print("pip install -r requirements.txt")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())