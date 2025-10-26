# Usage Guide

This guide provides detailed instructions for using the Facial Recognition System with AES-Encrypted Embeddings.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Command Line Interface](#command-line-interface)
4. [Python API](#python-api)
5. [Configuration](#configuration)
6. [Security Considerations](#security-considerations)
7. [Troubleshooting](#troubleshooting)

## Installation

### Automatic Installation

Run the installation script:

```bash
python install.py
```

### Manual Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Create data directories:
```bash
python examples.py --setup
```

3. Test the installation:
```bash
python test_system.py
```

## Quick Start

### 1. Prepare Sample Images

Place face images in the appropriate directories:
- `data/enrollment/` - Images for user enrollment
- `data/query/` - Images for authentication/identification
- `data/test/` - Images for testing and comparison

### 2. Enroll Users

```bash
# Enroll a single user
python main.py --mode enroll --user-id john_doe --image data/enrollment/john_face.jpg

# Enroll with metadata
python main.py --mode enroll --user-id jane_smith --image data/enrollment/jane_face.jpg --metadata '{"name": "Jane Smith", "department": "Engineering"}'
```

### 3. Authenticate Users

```bash
# Authenticate with default threshold
python main.py --mode authenticate --image data/query/query_face.jpg

# Authenticate with custom threshold
python main.py --mode authenticate --image data/query/query_face.jpg --threshold 0.7
```

### 4. Identify Users

```bash
# Identify user from image
python main.py --mode identify --image data/query/unknown_face.jpg

# Get top 10 matches
python main.py --mode identify --image data/query/unknown_face.jpg --threshold 0.5
```

### 5. Compare Faces

```bash
# Compare two face images
python main.py --mode compare --image data/test/face1.jpg --image2 data/test/face2.jpg
```

## Command Line Interface

### Main Script: `main.py`

The main script provides a command-line interface for all system operations.

#### General Options

- `--mode`: Operation mode (enroll, authenticate, identify, compare, stats)
- `--device`: Device to run models on (auto, cpu, cuda)
- `--password`: Password for encryption key derivation
- `--threshold`: Similarity threshold for matching (0.0-1.0)

#### Enrollment Mode

```bash
python main.py --mode enroll [OPTIONS]

Required:
  --user-id USER_ID    Unique identifier for the user
  --image IMAGE_PATH   Path to the user's face image

Optional:
  --metadata JSON      Additional metadata as JSON string
```

#### Authentication Mode

```bash
python main.py --mode authenticate [OPTIONS]

Required:
  --image IMAGE_PATH   Path to the query image

Optional:
  --threshold FLOAT    Similarity threshold (default: 0.6)
```

#### Identification Mode

```bash
python main.py --mode identify [OPTIONS]

Required:
  --image IMAGE_PATH   Path to the query image

Optional:
  --threshold FLOAT    Similarity threshold (default: 0.6)
```

#### Comparison Mode

```bash
python main.py --mode compare [OPTIONS]

Required:
  --image IMAGE_PATH1  Path to first image
  --image2 IMAGE_PATH2 Path to second image
```

#### Statistics Mode

```bash
python main.py --mode stats
```

## Python API

### Basic Usage

```python
from matching_module import FaceMatcher

# Initialize the system
matcher = FaceMatcher(device='auto')

# Enroll a user
result = matcher.enroll_user(
    user_id="john_doe",
    image_path="data/john_face.jpg",
    metadata={"name": "John Doe", "department": "Engineering"}
)

if result['success']:
    print(f"User enrolled: {result['user_id']}")
    print(f"Embeddings created: {result['embeddings_created']}")
else:
    print(f"Enrollment failed: {result['error']}")
```

### Authentication

```python
# Authenticate a user
result = matcher.authenticate_user(
    image_path="data/query_face.jpg",
    threshold=0.6
)

if result['success'] and result['authenticated']:
    print(f"User authenticated: {result['user_id']}")
    print(f"Similarity: {result['similarity']:.4f}")
else:
    print("Authentication failed")
```

### Identification

```python
# Identify a user
result = matcher.identify_user(
    image_path="data/query_face.jpg",
    threshold=0.5
)

if result['success']:
    print(f"Found {result['total_matches']} matches:")
    for match in result['matches']:
        print(f"  User: {match['user_id']}, Similarity: {match['similarity']:.4f}")
```

### Face Comparison

```python
# Compare two faces
result = matcher.compare_faces("data/face1.jpg", "data/face2.jpg")

if result['success']:
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"Same person: {result['same_person']}")
```

### User Management

```python
# Get user information
user_info = matcher.get_user_info("john_doe")
if user_info['success']:
    print(f"User has {user_info['total_embeddings']} embeddings")

# Delete user
delete_result = matcher.delete_user("john_doe")
if delete_result['success']:
    print(f"Deleted {delete_result['deleted_embeddings']} embeddings")

# Get system statistics
stats = matcher.get_system_stats()
if stats['success']:
    print(f"Total embeddings in system: {stats['total_embeddings']}")
```

## Configuration

### Configuration File: `config.py`

Key configuration parameters:

```python
# Face detection settings
FACE_SIZE = (160, 160)           # FaceNet input size
MARGIN = 20                      # Margin around detected face
MIN_FACE_SIZE = 20               # Minimum face size to detect

# Embedding settings
EMBEDDING_DIM = 512              # FaceNet embedding dimension
SIMILARITY_THRESHOLD = 0.6       # Default similarity threshold

# ChromaDB settings
CHROMA_COLLECTION_NAME = "face_embeddings"
CHROMA_PERSIST_DIRECTORY = "chroma_db"

# Encryption settings
AES_KEY_SIZE = 32                # 256-bit key
AES_IV_SIZE = 16                 # 128-bit IV
```

### Environment Variables

You can override configuration using environment variables:

```bash
export FACIAL_REC_SIMILARITY_THRESHOLD=0.7
export FACIAL_REC_DEVICE=cuda
export FACIAL_REC_CHROMA_DIR=/path/to/chroma
```

## Security Considerations

### Encryption

- All face embeddings are encrypted with AES-256 before storage
- Encryption keys are securely generated and stored
- No raw images or plain embeddings are permanently stored

### Key Management

- Default: Keys are stored in `encryption_key.key`
- Custom: Use `--password` option for password-based key derivation
- Production: Implement secure key management system

### Privacy

- Only encrypted embeddings are stored in ChromaDB
- Original images are not stored after processing
- Similarity search works on encrypted data

### Access Control

- Use appropriate similarity thresholds for your security requirements
- Implement additional authentication layers as needed
- Monitor and log access attempts

## Troubleshooting

### Common Issues

#### 1. "No faces detected in image"

**Solutions:**
- Ensure face is clearly visible and well-lit
- Try different image angles
- Check image quality and resolution
- Verify image format is supported (JPG, PNG)

#### 2. "Could not extract face embedding"

**Solutions:**
- Ensure face detection was successful
- Check if face region is properly cropped
- Verify image preprocessing is working
- Try with a different image

#### 3. "Authentication failed: No matching face found"

**Solutions:**
- Lower the similarity threshold
- Ensure user was properly enrolled
- Check if query image quality is good
- Verify encryption keys are consistent

#### 4. "CUDA out of memory"

**Solutions:**
- Use CPU device: `--device cpu`
- Process images one at a time
- Reduce batch sizes
- Close other GPU applications

#### 5. "ChromaDB connection error"

**Solutions:**
- Check if ChromaDB directory is writable
- Ensure no other processes are using the database
- Try resetting the collection
- Check disk space

### Debug Mode

Enable debug output by setting environment variable:

```bash
export FACIAL_REC_DEBUG=1
python main.py --mode authenticate --image data/query.jpg
```

### Logging

The system provides detailed logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your code here
```

### Performance Optimization

1. **Use GPU when available:**
   ```bash
   python main.py --device cuda --mode authenticate --image data/query.jpg
   ```

2. **Batch processing:**
   ```python
   # Process multiple images at once
   embeddings = matcher.face_embedder.extract_embeddings_batch(face_images)
   ```

3. **Optimize similarity threshold:**
   - Higher threshold = more secure, fewer false positives
   - Lower threshold = more permissive, more false positives

### Testing

Run comprehensive tests:

```bash
# Test all components
python test_system.py

# Test specific components
python test_system.py --component detection
python test_system.py --component encryption
```

### Getting Help

1. Check the logs for error messages
2. Run the test suite to verify installation
3. Try the example scripts
4. Review the configuration settings
5. Check system requirements and dependencies

## Advanced Usage

### Custom Face Detection

```python
from face_detection import FaceDetector

detector = FaceDetector(device='cpu')
faces = detector.detect_faces(image)
for face_info in faces:
    face = detector.extract_face(image, face_info)
    # Process face...
```

### Custom Embedding Extraction

```python
from face_embedding import FaceEmbedder

embedder = FaceEmbedder(device='cpu')
embedding = embedder.extract_embedding(face_image)
similarity = embedder.compute_similarity(embedding1, embedding2)
```

### Direct ChromaDB Access

```python
from chromadb_storage import ChromaDBStorage

storage = ChromaDBStorage()
similar_embeddings = storage.search_similar(
    query_encrypted_embedding=encrypted_data,
    n_results=10,
    threshold=0.6
)
```

### Encryption Management

```python
from encryption import EmbeddingEncryptor

encryptor = EmbeddingEncryptor(password="your_password")
encrypted_data, iv = encryptor.encrypt_embedding(embedding)
decrypted_embedding = encryptor.decrypt_embedding(encrypted_data, iv)
```
