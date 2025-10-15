"""
Script to create dummy dataset for testing the facial recognition system.
Downloads sample face images from public sources.
"""
import os
import urllib.request
import cv2
import numpy as np
from pathlib import Path

# Sample face image URLs (public domain or creative commons)
SAMPLE_IMAGES = [
    "https://picsum.photos/200/200?random=1",  # Random face-like image
    "https://picsum.photos/200/200?random=2",  # Random face-like image
    "https://picsum.photos/200/200?random=3",  # Random face-like image
    "https://picsum.photos/200/200?random=4",  # Random face-like image
]

def download_image(url, save_path):
    """Download image from URL and save to path."""
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
        with open(save_path, 'wb') as f:
            f.write(data)
        print(f"Downloaded: {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def create_synthetic_face(save_path):
    """Create a synthetic face image for testing."""
    # Create a blank image
    img = np.ones((200, 200, 3), dtype=np.uint8) * 200  # Light gray background

    # Draw simple face features
    # Face outline
    cv2.ellipse(img, (100, 100), (60, 80), 0, 0, 360, (139, 69, 19), 2)

    # Eyes
    cv2.circle(img, (85, 85), 8, (0, 0, 0), -1)
    cv2.circle(img, (115, 85), 8, (0, 0, 0), -1)

    # Nose
    cv2.ellipse(img, (100, 105), (3, 5), 0, 0, 360, (139, 69, 19), -1)

    # Mouth
    cv2.ellipse(img, (100, 125), (10, 5), 0, 0, 360, (0, 0, 0), 2)

    cv2.imwrite(save_path, img)
    print(f"Created synthetic face: {save_path}")

def create_dummy_dataset():
    """Create dummy dataset with sample images."""
    data_dir = Path("data")
    enrollment_dir = data_dir / "enrollment"
    query_dir = data_dir / "query"
    test_dir = data_dir / "test"

    # Create directories
    enrollment_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print("Creating dummy dataset...")

    # Download sample images for enrollment
    for i, url in enumerate(SAMPLE_IMAGES):
        filename = f"user_{i+1}.jpg"
        save_path = enrollment_dir / filename
        if download_image(url, str(save_path)):
            # Also copy to query for testing
            query_path = query_dir / f"query_{i+1}.jpg"
            cv2.imwrite(str(query_path), cv2.imread(str(save_path)))

    # Create some synthetic faces for additional testing
    for i in range(2):
        filename = f"synthetic_{i+1}.jpg"
        save_path = test_dir / filename
        create_synthetic_face(str(save_path))

    print("Dummy dataset created successfully!")
    print(f"Enrollment images: {len(list(enrollment_dir.glob('*.jpg')))}")
    print(f"Query images: {len(list(query_dir.glob('*.jpg')))}")
    print(f"Test images: {len(list(test_dir.glob('*.jpg')))}")

if __name__ == "__main__":
    create_dummy_dataset()
