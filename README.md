# Facial Recognition System with AES-Encrypted Embeddings

A secure facial recognition system that uses MTCNN for face detection, FaceNet for embedding extraction, AES encryption for privacy protection, and ChromaDB for efficient vector storage and similarity search.

## Features

- **Face Detection**: Robust face detection using MTCNN with landmark localization
- **Feature Extraction**: High-quality face embeddings using pre-trained FaceNet model
- **Privacy Protection**: AES-256 encryption for all stored face embeddings
- **Efficient Storage**: ChromaDB vector database for fast similarity search
- **Secure Authentication**: Privacy-preserving face matching and user authentication
- **Easy Integration**: Simple Python API for enrollment, authentication, and identification

## System Architecture

```
Input Image → MTCNN Detection → FaceNet Embedding → AES Encryption → ChromaDB Storage
                                                                    ↓
Query Image → MTCNN Detection → FaceNet Embedding → AES Encryption → Similarity Search
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd FacialRec
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Install in development mode:
```bash
pip install -e .
```

## Quick Start

### 1. Enroll a User
```bash
python main.py --mode enroll --user-id john_doe --image data/john_face.jpg
```

### 2. Authenticate a User
```bash
python main.py --mode authenticate --image data/query_face.jpg
```

### 3. Identify a User
```bash
python main.py --mode identify --image data/query_face.jpg
```

### 4. Compare Two Faces
```bash
python main.py --mode compare --image data/face1.jpg --image2 data/face2.jpg
```

### 5. View System Statistics
```bash
python main.py --mode stats
```

## Python API Usage

```python
from matching_module import FaceMatcher

# Initialize the system
matcher = FaceMatcher(device='auto')

# Enroll a new user
result = matcher.enroll_user(
    user_id="john_doe",
    image_path="data/john_face.jpg",
    metadata={"name": "John Doe", "department": "Engineering"}
)

# Authenticate a user
result = matcher.authenticate_user(
    image_path="data/query_face.jpg",
    threshold=0.6
)

# Identify a user
result = matcher.identify_user(
    image_path="data/query_face.jpg",
    threshold=0.5
)

# Compare two faces
result = matcher.compare_faces("data/face1.jpg", "data/face2.jpg")
```

## Configuration

Edit `config.py` to customize system settings:

- `SIMILARITY_THRESHOLD`: Minimum similarity score for face matching (default: 0.6)
- `FACE_SIZE`: Input size for face images (default: 160x160)
- `EMBEDDING_DIM`: FaceNet embedding dimension (default: 512)
- `CHROMA_PERSIST_DIRECTORY`: ChromaDB storage directory

## Security Features

- **AES-256 Encryption**: All face embeddings are encrypted before storage
- **Secure Key Management**: Encryption keys are securely generated and stored
- **Privacy-Preserving**: No raw images or plain embeddings are ever stored
- **Access Control**: Configurable similarity thresholds for authentication

## Testing

Run the test suite to verify system functionality:

```bash
# Run all tests
python test_system.py

# Test specific components
python test_system.py --component detection
python test_system.py --component embedding
python test_system.py --component encryption
python test_system.py --component storage
python test_system.py --component matching
```

## Examples

Run example scripts to see the system in action:

```bash
# Run all examples
python examples.py

# Run specific examples
python examples.py --enrollment
python examples.py --authentication
python examples.py --identification
python examples.py --comparison
python examples.py --management
```

## Project Structure

```
FacialRec/
├── main.py                 # Main command-line interface
├── config.py              # Configuration settings
├── face_detection.py      # MTCNN face detection
├── face_embedding.py      # FaceNet embedding extraction
├── encryption.py          # AES encryption utilities
├── chromadb_storage.py    # ChromaDB storage interface
├── matching_module.py     # Main matching and authentication
├── examples.py            # Example usage scripts
├── test_system.py         # Test suite
├── setup.py               # Package setup
├── requirements.txt       # Python dependencies
├── data/                  # Sample images directory
│   ├── enrollment/        # Images for user enrollment
│   ├── query/            # Images for authentication
│   └── test/             # Images for testing
└── README.md             # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- OpenCV 4.5+
- ChromaDB 0.4+
- MTCNN
- FaceNet-PyTorch
- Cryptography

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Support

For questions and support, please open an issue on GitHub or contact the development team.
A passion project for my APP project report
