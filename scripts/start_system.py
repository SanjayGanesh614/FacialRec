#!/usr/bin/env python3
"""
Startup script for the Facial Recognition System.
This script will install dependencies and start the web server.
"""
import subprocess
import sys
import os
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_tests():
    """Run system tests."""
    print("\n🧪 Running system tests...")
    try:
        result = subprocess.run([sys.executable, "extra/tests/test_setup.py"], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def start_web_server():
    """Start the web server."""
    print("\n🚀 Starting the Facial Recognition Web Server...")
    print("The server will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start the web server
        subprocess.run([sys.executable, "src/web/web_server.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

def main():
    """Main startup function."""
    print("🎭 Facial Recognition System Startup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found. Please run this script from the project root directory.")
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies. Please check your internet connection and try again.")
        return 1
    
    # Run tests
    if not run_tests():
        print("⚠️  Some tests failed, but continuing anyway...")
        print("You may experience issues with the system.")
        time.sleep(3)
    
    # Start web server
    start_web_server()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())