"""
Main pipeline for the facial recognition system.
"""
import argparse
import os
import sys
from pathlib import Path
from src.core.matching_module import FaceMatcher
from src.core import config


def main():
    """Main function for the facial recognition system."""
    parser = argparse.ArgumentParser(description='Facial Recognition System with AES-Encrypted Embeddings')
    parser.add_argument('--mode', choices=['enroll', 'authenticate', 'identify', 'compare', 'stats'], 
                       required=True, help='Operation mode')
    parser.add_argument('--user-id', type=str, help='User ID for enrollment')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--image2', type=str, help='Path to second image for comparison')
    parser.add_argument('--threshold', type=float, default=config.SIMILARITY_THRESHOLD,
                       help=f'Similarity threshold (default: {config.SIMILARITY_THRESHOLD})')
    parser.add_argument('--password', type=str, help='Password for encryption key derivation')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to run models on')
    
    args = parser.parse_args()
    
    # Initialize face matcher
    print("Initializing facial recognition system...")
    matcher = FaceMatcher(password=args.password, device=args.device)
    
    if args.mode == 'enroll':
        enroll_user(matcher, args)
    elif args.mode == 'authenticate':
        authenticate_user(matcher, args)
    elif args.mode == 'identify':
        identify_user(matcher, args)
    elif args.mode == 'compare':
        compare_faces(matcher, args)
    elif args.mode == 'stats':
        show_stats(matcher)


def enroll_user(matcher: FaceMatcher, args):
    """Enroll a new user."""
    if not args.user_id or not args.image:
        print("Error: --user-id and --image are required for enrollment")
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    print(f"Enrolling user: {args.user_id}")
    print(f"Processing image: {args.image}")
    
    result = matcher.enroll_user(
        user_id=args.user_id,
        image_path=args.image,
        metadata={'enrolled_at': str(Path(args.image).stat().st_mtime)}
    )
    
    if result['success']:
        print(f"✅ User enrolled successfully!")
        print(f"   User ID: {result['user_id']}")
        print(f"   Faces detected: {result['faces_detected']}")
        print(f"   Embeddings created: {result['embeddings_created']}")
        print(f"   Embedding IDs: {result['embedding_ids']}")
    else:
        print(f"❌ Enrollment failed: {result['error']}")


def authenticate_user(matcher: FaceMatcher, args):
    """Authenticate a user."""
    if not args.image:
        print("Error: --image is required for authentication")
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    print(f"Authenticating user with image: {args.image}")
    print(f"Using threshold: {args.threshold}")
    
    result = matcher.authenticate_user(
        image_path=args.image,
        threshold=args.threshold
    )
    
    if result['success']:
        if result['authenticated']:
            print(f"✅ Authentication successful!")
            print(f"   User ID: {result['user_id']}")
            print(f"   Similarity: {result['similarity']:.4f}")
            print(f"   Confidence: {result['confidence']:.4f}")
        else:
            print(f"❌ Authentication failed: {result.get('error', 'No matching face found')}")
            print(f"   Similarity: {result.get('similarity', 0):.4f}")
    else:
        print(f"❌ Authentication error: {result['error']}")


def identify_user(matcher: FaceMatcher, args):
    """Identify a user from their face."""
    if not args.image:
        print("Error: --image is required for identification")
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    print(f"Identifying user with image: {args.image}")
    print(f"Using threshold: {args.threshold}")
    
    result = matcher.identify_user(
        image_path=args.image,
        threshold=args.threshold
    )
    
    if result['success']:
        print(f"Found {result['total_matches']} matches:")
        for i, match in enumerate(result['matches'], 1):
            print(f"   {i}. User ID: {match['user_id']}")
            print(f"      Similarity: {match['similarity']:.4f}")
            print(f"      Distance: {match['distance']:.4f}")
    else:
        print(f"❌ Identification error: {result['error']}")


def compare_faces(matcher: FaceMatcher, args):
    """Compare two face images."""
    if not args.image or not args.image2:
        print("Error: --image and --image2 are required for comparison")
        return
    
    if not os.path.exists(args.image) or not os.path.exists(args.image2):
        print("Error: One or both image files not found")
        return
    
    print(f"Comparing faces:")
    print(f"   Image 1: {args.image}")
    print(f"   Image 2: {args.image2}")
    
    result = matcher.compare_faces(args.image, args.image2)
    
    if result['success']:
        print(f"✅ Comparison completed!")
        print(f"   Similarity: {result['similarity']:.4f}")
        print(f"   Same person: {result['same_person']}")
    else:
        print(f"❌ Comparison error: {result['error']}")


def show_stats(matcher: FaceMatcher):
    """Show system statistics."""
    print("System Statistics:")
    print("=" * 50)
    
    stats = matcher.get_system_stats()
    
    if stats['success']:
        print(f"Total embeddings: {stats['total_embeddings']}")
        print(f"Collection name: {stats['collection_name']}")
    else:
        print(f"❌ Error getting stats: {stats['error']}")


if __name__ == "__main__":
    main()
