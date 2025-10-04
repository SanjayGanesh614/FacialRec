"""
Configuration settings for the facial recognition system.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ENCRYPTED_EMBEDDINGS_DIR = PROJECT_ROOT / "encrypted_embeddings"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
ENCRYPTED_EMBEDDINGS_DIR.mkdir(exist_ok=True)

# Model settings
FACENET_MODEL_PATH = MODELS_DIR / "facenet_model.pth"
MTCNN_MODEL_PATH = MODELS_DIR / "mtcnn_model.pth"

# Face detection settings
FACE_SIZE = (160, 160)  # FaceNet input size
MARGIN = 20  # Margin around detected face
MIN_FACE_SIZE = 20  # Minimum face size to detect

# Embedding settings
EMBEDDING_DIM = 512  # FaceNet embedding dimension
SIMILARITY_THRESHOLD = 0.6  # Cosine similarity threshold for matching

# ChromaDB settings
CHROMA_COLLECTION_NAME = "face_embeddings"
CHROMA_PERSIST_DIRECTORY = str(PROJECT_ROOT / "chroma_db")

# Encryption settings
AES_KEY_SIZE = 32  # 256-bit key
AES_IV_SIZE = 16   # 128-bit IV

# Security settings
KEY_FILE_PATH = PROJECT_ROOT / "encryption_key.key"
