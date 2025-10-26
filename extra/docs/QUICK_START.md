# 🚀 Quick Start Guide

## One-Command Setup

Run this single command to install dependencies and start the system:

```bash
python start_system.py
```

## Manual Setup

If you prefer manual setup:

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the System**
   ```bash
   python test_setup.py
   ```

3. **Start the Web Server**
   ```bash
   python web_server.py
   ```

## Using the System

1. **Open your browser** and go to: `http://localhost:5000`

2. **Camera Features**:
   - Click "📷 Start Camera" to activate your webcam
   - Click "📸 Capture Photo" to take a picture
   - Or upload an image file instead

3. **Enroll a User**:
   - Go to "👤 Enroll User" tab
   - Enter a unique User ID
   - Take a photo or upload an image
   - Click "✅ Enroll User"

4. **Authenticate**:
   - Go to "🔐 Authenticate" tab
   - Take a photo or upload an image
   - Click "🔐 Authenticate"

5. **Identify**:
   - Go to "🔍 Identify" tab
   - Take a photo or upload an image
   - Click "🔍 Identify" to find all matches

6. **Compare Faces**:
   - Go to "⚖️ Compare Faces" tab
   - Upload two images
   - Click "⚖️ Compare Faces"

## Features

- ✅ **Real-time Camera Capture**: Use your webcam to capture photos
- ✅ **Face Detection**: Automatic face detection using MTCNN
- ✅ **Face Recognition**: High-accuracy recognition using FaceNet
- ✅ **AES Encryption**: All face embeddings are encrypted
- ✅ **Vector Database**: Fast similarity search using ChromaDB
- ✅ **Web Interface**: Modern, responsive web interface
- ✅ **Multiple Operations**: Enroll, authenticate, identify, and compare

## System Requirements

- Python 3.8+
- Webcam (for camera features)
- 4GB+ RAM recommended
- GPU optional (will use CPU if no GPU available)

## Troubleshooting

1. **Camera not working**: Make sure your browser has camera permissions
2. **Models downloading slowly**: First run may take time to download pre-trained models
3. **Memory issues**: Close other applications or use CPU mode
4. **Import errors**: Run `pip install -r requirements.txt` again

## Security

- All face embeddings are encrypted with AES-256
- No raw images are stored permanently
- Encryption keys are automatically generated and stored securely