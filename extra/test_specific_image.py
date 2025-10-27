#!/usr/bin/env python3
"""
Test authentication with the specific user image.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.core.matching_module import FaceMatcher
import time

def test_specific_image():
    """Test authentication with the specific user image."""
    print("🔐 Testing Authentication with Specific Image")
    print("=" * 60)
    
    # Initialize face matcher
    print("Initializing Face Recognition System...")
    matcher = FaceMatcher(device='auto')
    
    # Image path
    image_path = r"C:\Users\ASUS\OneDrive\Documents\GitHub\FacialRec\data\user_images\user001\enrollment_20251027_022045.jpg"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"✅ Image found: {image_path}")
    
    # Test 1: Enroll the user
    print("\n1. Enrolling user with the image...")
    enroll_result = matcher.enroll_user(
        user_id="test_user_001",
        image_path=image_path,
        metadata={"test": "specific_image_test"}
    )
    
    print(f"Enrollment result: {enroll_result}")
    
    if not enroll_result['success']:
        print("❌ Enrollment failed!")
        return
    
    print(f"✅ User enrolled successfully with {enroll_result['faces_detected']} faces detected")
    
    # Test 2: Authenticate with the same image
    print("\n2. Authenticating with the same image...")
    auth_result = matcher.authenticate_user(
        image_path=image_path,
        threshold=0.5  # Lower threshold for testing
    )
    
    print(f"Authentication result: {auth_result}")
    
    if auth_result['success'] and auth_result['authenticated']:
        print(f"✅ Authentication successful!")
        print(f"   User ID: {auth_result['user_id']}")
        print(f"   Similarity: {auth_result['similarity']:.4f}")
    else:
        print("❌ Authentication failed!")
        if 'error' in auth_result:
            print(f"   Error: {auth_result['error']}")
    
    # Test 3: Identify with the same image
    print("\n3. Identifying with the same image...")
    identify_result = matcher.identify_user(
        image_path=image_path,
        threshold=0.5  # Lower threshold for testing
    )
    
    print(f"Identification result: {identify_result}")
    
    if identify_result['success'] and identify_result['matches']:
        print(f"✅ Identification successful!")
        print(f"   Total matches: {identify_result['total_matches']}")
        for i, match in enumerate(identify_result['matches']):
            print(f"   Match {i+1}: {match['user_id']} (similarity: {match['similarity']:.4f})")
    else:
        print("❌ Identification failed!")
        if 'error' in identify_result:
            print(f"   Error: {identify_result['error']}")
    
    # Test 4: Get system stats
    print("\n4. System statistics...")
    stats = matcher.get_system_stats()
    print(f"System stats: {stats}")
    
    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    test_specific_image()