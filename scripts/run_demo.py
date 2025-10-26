#!/usr/bin/env python3
"""
Complete demo script for the FaceGuard AI Facial Recognition System.
This script demonstrates all system capabilities and opens the web interface.
"""
import subprocess
import sys
import time
import webbrowser
import threading
from pathlib import Path
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def print_banner():
    """Print the system banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    🎭 FaceGuard AI - Advanced Facial Recognition System     ║
    ║                                                              ║
    ║    ✨ Features:                                              ║
    ║    • Real-time Camera Capture                               ║
    ║    • Military-Grade AES-256 Encryption                      ║
    ║    • Advanced Face Detection & Recognition                   ║
    ║    • Vector Database Storage (ChromaDB)                      ║
    ║    • Modern Web Interface                                    ║
    ║    • User Enrollment & Authentication                        ║
    ║    • Face Identification & Comparison                        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_system():
    """Check if the system is ready."""
    print("🔍 Checking system status...")
    
    # Check if required files exist
    required_files = [
        'src/web/web_server.py',
        'src/core/matching_module.py',
        'src/core/face_detection.py',
        'src/core/face_embedding.py',
        'src/core/encryption.py',
        'src/core/chromadb_storage.py',
        'src/core/config.py',
        'src/web/templates/index.html'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All system files present")
    
    # Test imports
    try:
        import cv2
        import numpy as np
        import chromadb
        import cryptography
        import flask
        print("✅ All dependencies available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    
    return True

def start_web_server():
    """Start the web server in a separate process."""
    print("🚀 Starting FaceGuard AI Web Server...")
    
    try:
        # Start the server
        process = subprocess.Popen(
            [sys.executable, "src/web/web_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        if process.poll() is None:
            print("✅ Web server started successfully!")
            print("🌐 Server running at: http://localhost:5000")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Server failed to start:")
            print(f"   stdout: {stdout}")
            print(f"   stderr: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None

def open_browser():
    """Open the web interface in the default browser."""
    print("🌐 Opening web interface...")
    
    # Wait a moment for server to be fully ready
    time.sleep(2)
    
    try:
        webbrowser.open('http://localhost:5000')
        print("✅ Web interface opened in your default browser")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("   Please manually open: http://localhost:5000")

def show_usage_guide():
    """Show usage guide."""
    guide = """
    📋 USAGE GUIDE
    ═══════════════════════════════════════════════════════════════
    
    🎯 Getting Started:
    1. The web interface should now be open in your browser
    2. If not, go to: http://localhost:5000
    
    👤 Enroll a User:
    • Click "Enroll User" tab
    • Click "Start Camera" to use your webcam
    • Click "Capture Photo" to take a picture
    • Enter a unique User ID
    • Click "Enroll User"
    
    🔐 Authenticate:
    • Click "Authenticate" tab
    • Take a photo or upload an image
    • Click "Authenticate" to verify identity
    
    🔍 Identify:
    • Click "Identify" tab
    • Take a photo or upload an image
    • Click "Identify" to find all matching users
    
    ⚖️ Compare Faces:
    • Click "Compare" tab
    • Upload two face images
    • Click "Compare Faces" to see similarity
    
    🛡️ Security Features:
    • All face data is encrypted with AES-256
    • No raw images are stored permanently
    • Vector database for fast similarity search
    • Privacy-preserving face matching
    
    ═══════════════════════════════════════════════════════════════
    """
    print(guide)

def monitor_server(process):
    """Monitor the server process."""
    try:
        while True:
            if process.poll() is not None:
                print("\n⚠️  Server process ended")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        process.terminate()
        process.wait()
        print("✅ Server stopped")

def main():
    """Main demo function."""
    print_banner()
    
    # Check system
    if not check_system():
        print("\n❌ System check failed. Please run 'python test_setup.py' first.")
        return 1
    
    print("✅ System check passed!")
    
    # Start web server
    server_process = start_web_server()
    if not server_process:
        print("\n❌ Failed to start web server")
        return 1
    
    # Open browser
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Show usage guide
    show_usage_guide()
    
    print("🎉 FaceGuard AI is now running!")
    print("Press Ctrl+C to stop the server")
    print("=" * 65)
    
    # Monitor server
    try:
        monitor_server(server_process)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        server_process.terminate()
        server_process.wait()
        print("✅ FaceGuard AI stopped successfully")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())