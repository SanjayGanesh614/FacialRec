# 🚀 FaceGuard AI - Installation Guide

## 📋 Prerequisites

- **Python 3.8+** (3.9 or 3.10 recommended)
- **Webcam** (for camera features)
- **4GB+ RAM** (8GB recommended)
- **Windows/Linux/macOS**

## ⚡ Quick Installation

### 1. Clone or Download
```bash
git clone <repository-url>
cd FacialRec
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch System
```bash
python launch.py
```

### 4. Open Web Interface
```
http://localhost:5000
```

## 🎯 Launch Options

### Option 1: Interactive Launcher
```bash
python launch.py
```
Choose from menu:
1. 🎯 Demo Mode (Full system)
2. 🌐 Web Server Only
3. 💻 CLI Mode
4. 🧪 Test Mode

### Option 2: Direct Launch
```bash
# Full demo with browser
python scripts/run_demo.py

# Web server only
python src/web/web_server.py

# System tests
python extra/tests/test_setup.py
```

## 🔧 Troubleshooting

### Common Issues

#### Dependencies
```bash
# If pip install fails
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

#### Camera Access
- Allow camera permissions in browser
- Close other applications using camera
- Try different browser if issues persist

#### Port Already in Use
```bash
# Kill process on port 5000
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

## ✅ Verification

After installation, verify system:
```bash
python extra/tests/test_setup.py
```

Should show:
```
🎉 All tests passed! The system is ready to use.
```

## 🎉 You're Ready!

System is now installed and ready to use. Launch with:
```bash
python launch.py
```

Then visit: `http://localhost:5000`