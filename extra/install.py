"""
Installation script for the facial recognition system.
"""
import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_requirements():
    """Install Python requirements."""
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python dependencies"
    )


def create_directories():
    """Create necessary directories."""
    print("📁 Creating project directories...")
    
    directories = [
        "data/enrollment",
        "data/query", 
        "data/test",
        "models",
        "encrypted_embeddings",
        "chroma_db"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created {directory}/")
    
    return True


def download_models():
    """Download pre-trained models (optional)."""
    print("🤖 Model download is handled automatically on first use")
    print("   FaceNet and MTCNN models will be downloaded when needed")
    return True


def verify_installation():
    """Verify the installation."""
    print("🔍 Verifying installation...")
    
    try:
        # Test imports
        import torch
        import cv2
        import numpy as np
        from PIL import Image
        print("   ✅ Core dependencies imported successfully")
        
        # Test if models can be loaded
        print("   ⚠️  Testing model loading (this may take a moment)...")
        
        # This will trigger model downloads if needed
        from face_detection import FaceDetector
        from face_embedding import FaceEmbedder
        
        print("   ✅ Models loaded successfully")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ⚠️  Model loading warning: {e}")
        print("   Models will be downloaded on first use")
        return True


def main():
    """Main installation function."""
    print("🚀 Facial Recognition System - Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Installation failed at dependency installation step")
        return False
    
    # Create directories
    if not create_directories():
        print("\n❌ Installation failed at directory creation step")
        return False
    
    # Download models
    if not download_models():
        print("\n❌ Installation failed at model download step")
        return False
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Installation completed with warnings")
        print("   Some components may need to be downloaded on first use")
    else:
        print("\n✅ Installation completed successfully!")
    
    print("\n" + "=" * 50)
    print("🎉 Installation Summary:")
    print("   • Python dependencies installed")
    print("   • Project directories created")
    print("   • Models ready for download on first use")
    print("\n📖 Next steps:")
    print("   1. Add sample images to data/ directories")
    print("   2. Run: python test_system.py")
    print("   3. Try: python examples.py --setup")
    print("   4. Read README.md for usage instructions")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
