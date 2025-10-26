#!/usr/bin/env python3
"""
FaceGuard AI - Quick Launch Script

This script provides a simple way to launch the FaceGuard AI system.
"""

import sys
import os
import subprocess
from pathlib import Path

def print_banner():
    """Print the FaceGuard AI banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║    🎭 FaceGuard AI - Advanced Facial Recognition System     ║
    ║                                                              ║
    ║    🚀 Quick Launch Options:                                  ║
    ║    1. Demo Mode - Full system demonstration                  ║
    ║    2. Web Server - Start web interface only                  ║
    ║    3. CLI Mode - Command line interface                      ║
    ║    4. Test Mode - Run system tests                           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main launcher function."""
    print_banner()
    
    print("Select launch mode:")
    print("1. 🎯 Demo Mode (Recommended)")
    print("2. 🌐 Web Server Only")
    print("3. 💻 CLI Mode")
    print("4. 🧪 Test Mode")
    print("5. ❌ Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                print("\n🚀 Launching Demo Mode...")
                subprocess.run([sys.executable, "scripts/run_demo.py"])
                break
                
            elif choice == '2':
                print("\n🌐 Starting Web Server...")
                subprocess.run([sys.executable, "src/web/web_server.py"])
                break
                
            elif choice == '3':
                print("\n💻 Starting CLI Mode...")
                subprocess.run([sys.executable, "extra/main.py", "--help"])
                break
                
            elif choice == '4':
                print("\n🧪 Running System Tests...")
                subprocess.run([sys.executable, "extra/tests/test_setup.py"])
                break
                
            elif choice == '5':
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            break

if __name__ == "__main__":
    main()