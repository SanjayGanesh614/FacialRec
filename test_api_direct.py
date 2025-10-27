#!/usr/bin/env python3
"""
Test the API endpoints directly to debug the web interface issues.
"""
import requests
import json
import os

def test_api_endpoints():
    """Test API endpoints directly."""
    base_url = "http://localhost:5000"
    
    print("🔍 Testing FaceGuard AI API Endpoints")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/api/health", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: System stats
    print("\n2. Testing stats endpoint...")
    try:
        response = requests.get(f"{base_url}/api/stats", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Enroll with existing image
    print("\n3. Testing enroll endpoint with existing image...")
    image_path = r"C:\Users\ASUS\OneDrive\Documents\GitHub\FacialRec\data\user_images\user001\enrollment_20251027_022045.jpg"
    
    if os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {'user_id': 'test_user_web'}
                
                print(f"   Sending request to {base_url}/api/enroll...")
                response = requests.post(f"{base_url}/api/enroll", files=files, data=data, timeout=30)
                print(f"   Status: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print(f"   ❌ Image not found: {image_path}")
    
    # Test 4: Authenticate with same image
    print("\n4. Testing authenticate endpoint...")
    if os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                data = {'threshold': '0.6'}
                
                print(f"   Sending request to {base_url}/api/authenticate...")
                response = requests.post(f"{base_url}/api/authenticate", files=files, data=data, timeout=30)
                print(f"   Status: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_api_endpoints()