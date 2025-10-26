#!/usr/bin/env python3
"""
Direct test of the matching functionality without web server.
"""
import sys
import os
import numpy as np
import cv2

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core.matching_module import FaceMatcher

def create_test_face_image(variation=0):
    """Create a more realistic test face image that OpenCV can detect."""
    # Create a larger image for better detection
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = (220, 220, 220)  # Light background
    
    center_x = 150
    center_y = 150
    
    # Face outline (larger and more defined)
    cv2.ellipse(img, (center_x, center_y), (100, 120), 0, 0, 360, (200, 170, 150), -1)
    cv2.ellipse(img, (center_x, center_y), (100, 120), 0, 0, 360, (180, 150, 130), 2)
    
    # Forehead
    cv2.ellipse(img, (center_x, center_y-40), (80, 60), 0, 0, 180, (210, 180, 160), -1)
    
    # Eyes (more prominent)
    eye_y = center_y - 20
    # Left eye
    cv2.ellipse(img, (center_x-35, eye_y), (18, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (center_x-35, eye_y), 8, (100, 80, 60), -1)
    cv2.circle(img, (center_x-35, eye_y), 4, (0, 0, 0), -1)
    
    # Right eye
    cv2.ellipse(img, (center_x+35, eye_y), (18, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (center_x+35, eye_y), 8, (100, 80, 60), -1)
    cv2.circle(img, (center_x+35, eye_y), 4, (0, 0, 0), -1)
    
    # Eyebrows
    cv2.ellipse(img, (center_x-35, eye_y-15), (20, 8), 0, 0, 180, (120, 100, 80), -1)
    cv2.ellipse(img, (center_x+35, eye_y-15), (20, 8), 0, 0, 180, (120, 100, 80), -1)
    
    # Nose (more defined)
    nose_y = center_y + 10
    cv2.line(img, (center_x, center_y), (center_x, nose_y+20), (160, 130, 110), 3)
    cv2.ellipse(img, (center_x, nose_y+20), (12, 6), 0, 0, 180, (160, 130, 110), 2)
    
    # Mouth (more prominent)
    mouth_y = center_y + 50
    cv2.ellipse(img, (center_x, mouth_y), (25, 12), 0, 0, 180, (140, 100, 100), 3)
    
    # Add some variation based on parameter
    if variation > 0:
        # Add slight rotation or shift
        M = cv2.getRotationMatrix2D((center_x, center_y), variation * 2, 1.0)
        img = cv2.warpAffine(img, M, (300, 300))
    
    # Add some noise for realism
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

def test_direct_matching():
    """Test the matching functionality directly."""
    print("🧪 Direct Matching Test")
    print("=" * 40)
    
    # Initialize matcher
    print("1. Initializing FaceMatcher...")
    matcher = FaceMatcher(device='cpu')
    
    # Create test images
    print("2. Creating test images...")
    enrollment_image = create_test_face_image(0)
    auth_image = create_test_face_image(2)
    
    # Save images temporarily
    cv2.imwrite("temp_enroll.jpg", enrollment_image)
    cv2.imwrite("temp_auth.jpg", auth_image)
    
    try:
        # Test enrollment
        print("3. Testing enrollment...")
        enroll_result = matcher.enroll_user(
            user_id="direct_test_user",
            image_path="temp_enroll.jpg",
            save_image=False
        )
        
        print(f"   Enrollment success: {enroll_result.get('success')}")
        if not enroll_result.get('success'):
            print(f"   Error: {enroll_result.get('error')}")
            return False
        
        print(f"   Faces detected: {enroll_result.get('faces_detected')}")
        print(f"   Embeddings created: {enroll_result.get('embeddings_created')}")
        
        # Test authentication
        print("4. Testing authentication...")
        auth_result = matcher.authenticate_user(
            image_path="temp_auth.jpg",
            threshold=0.1  # Very low threshold
        )
        
        print(f"   Authentication success: {auth_result.get('success')}")
        print(f"   Authenticated: {auth_result.get('authenticated')}")
        print(f"   User ID: {auth_result.get('user_id')}")
        print(f"   Similarity: {auth_result.get('similarity', 0):.4f}")
        
        if auth_result.get('error'):
            print(f"   Error: {auth_result.get('error')}")
        
        # Test identification
        print("5. Testing identification...")
        identify_result = matcher.identify_user(
            image_path="temp_auth.jpg",
            threshold=0.1
        )
        
        print(f"   Identification success: {identify_result.get('success')}")
        print(f"   Total matches: {identify_result.get('total_matches', 0)}")
        
        matches = identify_result.get('matches', [])
        for i, match in enumerate(matches[:3]):
            print(f"   Match {i+1}: {match.get('user_id')} (similarity: {match.get('similarity', 0):.4f})")
        
        # Check system stats
        print("6. System statistics...")
        stats = matcher.get_system_stats()
        print(f"   Total embeddings: {stats.get('total_embeddings', 0)}")
        
        return auth_result.get('authenticated', False)
        
    finally:
        # Cleanup
        try:
            os.remove("temp_enroll.jpg")
            os.remove("temp_auth.jpg")
        except:
            pass

def main():
    """Run the direct matching test."""
    try:
        success = test_direct_matching()
        
        print("\n" + "=" * 40)
        if success:
            print("🎉 Direct matching test PASSED!")
            print("Authentication and identification are working!")
        else:
            print("❌ Direct matching test FAILED!")
            print("Authentication/identification need more work.")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())