"""
Example usage scripts for the facial recognition system.
"""
import os
import sys
from pathlib import Path
from matching_module import FaceMatcher
import config


def example_enrollment():
    """Example of enrolling users."""
    print("=== Enrollment Example ===")
    
    # Initialize matcher
    matcher = FaceMatcher(device='auto')
    
    # Example user data
    users = [
        {
            'user_id': 'john_doe',
            'image_path': 'data/john_face1.jpg',
            'metadata': {'name': 'John Doe', 'department': 'Engineering'}
        },
        {
            'user_id': 'jane_smith',
            'image_path': 'data/jane_face1.jpg',
            'metadata': {'name': 'Jane Smith', 'department': 'Marketing'}
        }
    ]
    
    for user in users:
        print(f"\nEnrolling {user['user_id']}...")
        
        # Check if image exists
        if not os.path.exists(user['image_path']):
            print(f"⚠️  Image not found: {user['image_path']}")
            print("   Please add sample images to the data/ directory")
            continue
        
        result = matcher.enroll_user(
            user_id=user['user_id'],
            image_path=user['image_path'],
            metadata=user['metadata']
        )
        
        if result['success']:
            print(f"✅ {user['user_id']} enrolled successfully")
            print(f"   Faces detected: {result['faces_detected']}")
            print(f"   Embeddings created: {result['embeddings_created']}")
        else:
            print(f"❌ Failed to enroll {user['user_id']}: {result['error']}")


def example_authentication():
    """Example of authenticating users."""
    print("\n=== Authentication Example ===")
    
    # Initialize matcher
    matcher = FaceMatcher(device='auto')
    
    # Test images
    test_images = [
        'data/john_face2.jpg',
        'data/jane_face2.jpg',
        'data/unknown_face.jpg'
    ]
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"⚠️  Image not found: {image_path}")
            continue
        
        print(f"\nAuthenticating with {image_path}...")
        
        result = matcher.authenticate_user(
            image_path=image_path,
            threshold=0.6
        )
        
        if result['success']:
            if result['authenticated']:
                print(f"✅ Authentication successful!")
                print(f"   User: {result['user_id']}")
                print(f"   Similarity: {result['similarity']:.4f}")
            else:
                print(f"❌ Authentication failed: {result.get('error', 'No match found')}")
        else:
            print(f"❌ Authentication error: {result['error']}")


def example_identification():
    """Example of identifying users."""
    print("\n=== Identification Example ===")
    
    # Initialize matcher
    matcher = FaceMatcher(device='auto')
    
    # Test image
    test_image = 'data/query_face.jpg'
    
    if not os.path.exists(test_image):
        print(f"⚠️  Image not found: {test_image}")
        return
    
    print(f"Identifying user in {test_image}...")
    
    result = matcher.identify_user(
        image_path=test_image,
        threshold=0.5
    )
    
    if result['success']:
        print(f"Found {result['total_matches']} matches:")
        for i, match in enumerate(result['matches'], 1):
            print(f"   {i}. User: {match['user_id']}")
            print(f"      Similarity: {match['similarity']:.4f}")
            print(f"      Metadata: {match['metadata']}")
    else:
        print(f"❌ Identification error: {result['error']}")


def example_face_comparison():
    """Example of comparing two faces."""
    print("\n=== Face Comparison Example ===")
    
    # Initialize matcher
    matcher = FaceMatcher(device='auto')
    
    # Test image pairs
    image_pairs = [
        ('data/john_face1.jpg', 'data/john_face2.jpg'),
        ('data/jane_face1.jpg', 'data/jane_face2.jpg'),
        ('data/john_face1.jpg', 'data/jane_face1.jpg')
    ]
    
    for img1, img2 in image_pairs:
        if not os.path.exists(img1) or not os.path.exists(img2):
            print(f"⚠️  One or both images not found: {img1}, {img2}")
            continue
        
        print(f"\nComparing {Path(img1).name} vs {Path(img2).name}...")
        
        result = matcher.compare_faces(img1, img2)
        
        if result['success']:
            print(f"   Similarity: {result['similarity']:.4f}")
            print(f"   Same person: {result['same_person']}")
        else:
            print(f"❌ Comparison error: {result['error']}")


def example_user_management():
    """Example of user management operations."""
    print("\n=== User Management Example ===")
    
    # Initialize matcher
    matcher = FaceMatcher(device='auto')
    
    # Get user info
    user_id = 'john_doe'
    print(f"Getting info for user: {user_id}")
    
    user_info = matcher.get_user_info(user_id)
    if user_info['success']:
        print(f"   Total embeddings: {user_info['total_embeddings']}")
        for i, emb in enumerate(user_info['embeddings']):
            print(f"   Embedding {i+1}: {emb['id']}")
    else:
        print(f"❌ Error getting user info: {user_info['error']}")
    
    # Show system stats
    print(f"\nSystem Statistics:")
    stats = matcher.get_system_stats()
    if stats['success']:
        print(f"   Total embeddings: {stats['total_embeddings']}")
        print(f"   Collection: {stats['collection_name']}")
    else:
        print(f"❌ Error getting stats: {stats['error']}")


def create_sample_data_structure():
    """Create sample data directory structure."""
    print("\n=== Creating Sample Data Structure ===")
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Create sample subdirectories
    sample_dirs = ['enrollment', 'query', 'test']
    for subdir in sample_dirs:
        (data_dir / subdir).mkdir(exist_ok=True)
    
    print("Created data directory structure:")
    print("   data/")
    print("   ├── enrollment/  (place enrollment images here)")
    print("   ├── query/       (place query images here)")
    print("   └── test/        (place test images here)")
    
    # Create README for data directory
    readme_content = """# Data Directory

This directory contains sample images for the facial recognition system.

## Directory Structure

- `enrollment/` - Images for enrolling new users
- `query/` - Images for authentication and identification
- `test/` - Images for testing and comparison

## Image Requirements

- Supported formats: JPG, JPEG, PNG
- Recommended size: At least 160x160 pixels
- Face should be clearly visible and well-lit
- Avoid extreme angles or occlusions

## Sample Usage

1. Place enrollment images in `enrollment/` directory
2. Use images from `query/` for authentication
3. Test with images in `test/` directory
"""
    
    with open(data_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print("Created data/README.md with usage instructions")


def main():
    """Run all examples."""
    print("Facial Recognition System - Examples")
    print("=" * 50)
    
    # Create sample data structure
    create_sample_data_structure()
    
    # Run examples
    try:
        example_enrollment()
        example_authentication()
        example_identification()
        example_face_comparison()
        example_user_management()
        
        print("\n" + "=" * 50)
        print("Examples completed!")
        print("\nTo run individual examples, you can call:")
        print("  python examples.py --enrollment")
        print("  python examples.py --authentication")
        print("  python examples.py --identification")
        print("  python examples.py --comparison")
        print("  python examples.py --management")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have installed all requirements:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Facial Recognition Examples')
    parser.add_argument('--enrollment', action='store_true', help='Run enrollment example')
    parser.add_argument('--authentication', action='store_true', help='Run authentication example')
    parser.add_argument('--identification', action='store_true', help='Run identification example')
    parser.add_argument('--comparison', action='store_true', help='Run face comparison example')
    parser.add_argument('--management', action='store_true', help='Run user management example')
    parser.add_argument('--setup', action='store_true', help='Create sample data structure')
    
    args = parser.parse_args()
    
    if args.setup:
        create_sample_data_structure()
    elif args.enrollment:
        example_enrollment()
    elif args.authentication:
        example_authentication()
    elif args.identification:
        example_identification()
    elif args.comparison:
        example_face_comparison()
    elif args.management:
        example_user_management()
    else:
        main()
