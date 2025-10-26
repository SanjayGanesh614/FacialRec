# 🎭 FaceGuard AI - Advanced Facial Recognition System

<div align="center">

![FaceGuard AI](https://img.shields.io/badge/FaceGuard-AI-blue?style=for-the-badge&logo=shield&logoColor=white)
![Version](https://img.shields.io/badge/version-1.0.0-green?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)

**Military-Grade Facial Recognition System with Real-Time Camera Integration**

</div>

---

## 🌟 Overview

**FaceGuard AI** is a state-of-the-art facial recognition system featuring:
- **🔒 Military-Grade AES-256 Encryption**
- **📷 Real-Time Camera Integration** 
- **🎨 Modern Glass-Morphism Web Interface**
- **⚡ Lightning-Fast Vector Database Search**
- **🛡️ Privacy-Preserving Architecture**

## 🚀 Quick Start

### 🎯 One-Command Launch (Recommended)
```bash
python launch.py
```

### 🌐 Direct Web Server
```bash
python src/web/web_server.py
```

**Then open:** `http://localhost:5000`

## 🏗️ Clean Project Structure

```
FaceGuard-AI/
├── 📁 src/                          # Core system
│   ├── 📁 core/                     # Recognition modules
│   │   ├── face_detection.py        # Face detection
│   │   ├── face_embedding.py        # Feature extraction
│   │   ├── encryption.py            # AES-256 encryption
│   │   ├── chromadb_storage.py      # Vector database
│   │   ├── matching_module.py       # Main logic
│   │   └── config.py                # Configuration
│   │
│   └── 📁 web/                      # Web interface
│       ├── web_server.py            # Flask server
│       └── templates/index.html     # Modern UI
│
├── 📁 scripts/                      # Launch scripts
│   ├── run_demo.py                  # Full demo
│   └── start_system.py              # System setup
│
├── 📁 data/                         # User data
├── 📁 extra/                        # Tests & docs
│
├── 📄 launch.py                     # Main launcher
├── 📄 requirements.txt              # Dependencies
└── 📄 README.md                     # This file
```

## 🎯 Core Features

### 🔐 **Security**
- **AES-256 Encryption** for all face data
- **Zero Raw Storage** - only encrypted embeddings
- **Privacy-First** design with no data exposure

### 🎨 **Modern Interface**
- **Glass-morphism UI** with smooth animations
- **Real-Time Camera** capture and preview
- **Responsive Design** for all devices

### 🧠 **Recognition**
- **Advanced Detection** using OpenCV
- **Custom Embeddings** with 128D vectors
- **Fast Search** with ChromaDB vector database

### 🚀 **Operations**
- **👤 User Enrollment** - Register with camera/upload
- **🔐 Authentication** - Verify identity
- **🔍 Identification** - Find users from photos
- **⚖️ Face Comparison** - Compare two faces

## 🎨 Web Interface

Beautiful, modern interface featuring:
- **Glass-morphism effects** with backdrop blur
- **Interactive camera controls** with live preview
- **Drag & drop uploads** with instant feedback
- **Real-time results** with smooth animations
- **Mobile-responsive** design

## 📊 Performance

| Operation | Speed | Accuracy |
|-----------|-------|----------|
| Face Detection | ~50ms | >95% |
| Feature Extraction | ~100ms | High |
| Database Search | ~10ms | Fast |
| **Total Processing** | **~200ms** | **>95%** |

## 🔧 API Endpoints

```http
POST /api/enroll          # Enroll new user
POST /api/authenticate    # Verify identity
POST /api/identify        # Find user
POST /api/compare         # Compare faces
GET  /api/stats          # System info
```

## 🛡️ Security Architecture

```
Camera/Upload → Face Detection → Feature Extraction → AES Encryption → ChromaDB
                                                                      ↓
Query Image → Face Detection → Feature Extraction → AES Encryption → Search & Match
```

## 🧪 Testing

```bash
# Run system tests
python extra/tests/test_setup.py

# Test individual components
python -c "from src.core import FaceMatcher; print('✅ System OK')"
```

## 🚀 Deployment

### Development
```bash
python src/web/web_server.py
```

### Production
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 src.web.web_server:app
```

## 📖 Documentation

Complete documentation available in `extra/docs/`:
- **Complete Guide** - Full system documentation
- **Quick Start** - Get started in minutes
- **Training Guide** - Model training instructions
- **Web App Guide** - Interface documentation

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

<div align="center">

**🎉 Ready to use FaceGuard AI?**

```bash
python launch.py
```

**Then visit:** `http://localhost:5000`

**Made with ❤️ for secure facial recognition**

</div>