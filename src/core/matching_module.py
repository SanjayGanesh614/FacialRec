"""
Matching module for face recognition and authentication.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from .face_detection import FaceDetector
from .face_embedding import FaceEmbedder
from .encryption import EmbeddingEncryptor
from .chromadb_storage import ChromaDBStorage
from . import config
import cv2
import base64
import io
from PIL import Image


class FaceMatcher:
    """Face matching and authentication module."""
    
    def __init__(self, password: Optional[str] = None, device: str = 'auto'):
        """
        Initialize the face matcher.
        
        Args:
            password: Password for encryption key derivation
            device: Device to run models on ('cpu', 'cuda', or 'auto')
        """
        print("Initializing Face Recognition System...")
        self.face_detector = FaceDetector(device=device)
        self.face_embedder = FaceEmbedder(device=device)
        self.encryptor = EmbeddingEncryptor(password=password)
        self.storage = ChromaDBStorage()
        print("Face Recognition System initialized successfully!")
    
    def enroll_user(self, user_id: str, image_path: str = None, image_data: bytes = None, metadata: Dict = None, save_image: bool = True) -> Dict:
        """
        Enroll a new user by processing their face image.
        
        Args:
            user_id: Unique identifier for the user
            image_path: Path to the user's face image (optional if image_data provided)
            image_data: Raw image data as bytes (optional if image_path provided)
            metadata: Additional metadata to store
            
        Returns:
            Dictionary with enrollment results
        """
        try:
            # Load image from path or data
            if image_path:
                image = self.face_detector.load_image(image_path)
                original_image_path = image_path
            elif image_data:
                image = self._decode_image_data(image_data)
                original_image_path = None
            else:
                return {
                    'success': False,
                    'error': 'Either image_path or image_data must be provided',
                    'user_id': user_id
                }
            
            if image is None:
                return {
                    'success': False,
                    'error': 'Could not load image',
                    'user_id': user_id
                }
            
            # Save original image if requested
            saved_image_path = None
            if save_image:
                saved_image_path = self._save_user_image(user_id, image, 'enrollment')
            
            # Detect and extract faces
            faces = self.face_detector.extract_faces_from_image(image)
            if not faces:
                return {
                    'success': False,
                    'error': 'No faces detected in image',
                    'user_id': user_id
                }
            
            # Extract embeddings for all detected faces
            embeddings = self.face_embedder.extract_embeddings_batch(faces)
            valid_embeddings = [emb for emb in embeddings if emb is not None]
            
            if not valid_embeddings:
                return {
                    'success': False,
                    'error': 'Could not extract face embeddings',
                    'user_id': user_id
                }
            
            # Encrypt embeddings
            encrypted_embeddings = []
            for embedding in valid_embeddings:
                encrypted_data, iv = self.encryptor.encrypt_embedding(embedding)
                encrypted_embeddings.append(encrypted_data)
            
            # Store in ChromaDB
            embedding_ids = self.storage.add_embeddings_batch(
                user_ids=[user_id] * len(encrypted_embeddings),
                encrypted_embeddings=encrypted_embeddings,
                metadatas=[metadata] * len(encrypted_embeddings) if metadata else None
            )
            
            return {
                'success': True,
                'user_id': user_id,
                'faces_detected': len(faces),
                'embeddings_created': len(valid_embeddings),
                'embedding_ids': embedding_ids,
                'saved_image_path': saved_image_path,
                'face_locations': [face_info.get('box', []) for face_info in self.face_detector.detect_faces(image)]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Enrollment failed: {str(e)}',
                'user_id': user_id
            }
    
    def authenticate_user(self, image_path: str = None, image_data: bytes = None, threshold: float = None) -> Dict:
        """
        Authenticate a user by comparing their face with stored embeddings.
        
        Args:
            image_path: Path to the query image (optional if image_data provided)
            image_data: Raw image data as bytes (optional if image_path provided)
            threshold: Similarity threshold for matching (uses config default if None)
            
        Returns:
            Dictionary with authentication results
        """
        try:
            # Load image from path or data
            if image_path:
                image = self.face_detector.load_image(image_path)
            elif image_data:
                image = self._decode_image_data(image_data)
            else:
                return {
                    'success': False,
                    'error': 'Either image_path or image_data must be provided',
                    'authenticated': False
                }
            
            if image is None:
                return {
                    'success': False,
                    'error': 'Could not load image',
                    'authenticated': False
                }
            
            # Detect and extract faces
            faces = self.face_detector.extract_faces_from_image(image)
            if not faces:
                return {
                    'success': False,
                    'error': 'No faces detected in image',
                    'authenticated': False
                }
            
            # Use the first detected face for authentication
            face = faces[0]
            
            # Extract embedding
            embedding = self.face_embedder.extract_embedding(face)
            if embedding is None:
                return {
                    'success': False,
                    'error': 'Could not extract face embedding',
                    'authenticated': False
                }
            
            # Encrypt query embedding
            encrypted_data, iv = self.encryptor.encrypt_embedding(embedding)
            
            # Search for similar embeddings
            threshold = threshold or config.SIMILARITY_THRESHOLD
            similar_embeddings = self.storage.search_similar(
                query_encrypted_embedding=encrypted_data,
                n_results=5,
                threshold=threshold
            )
            
            if similar_embeddings:
                best_match = similar_embeddings[0]
                return {
                    'success': True,
                    'authenticated': True,
                    'user_id': best_match['user_id'],
                    'similarity': best_match['similarity'],
                    'confidence': best_match['similarity'],
                    'all_matches': similar_embeddings
                }
            else:
                return {
                    'success': True,
                    'authenticated': False,
                    'error': 'No matching face found',
                    'similarity': 0.0
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Authentication failed: {str(e)}',
                'authenticated': False
            }
    
    def identify_user(self, image_path: str = None, image_data: bytes = None, threshold: float = None) -> Dict:
        """
        Identify a user from their face (returns all possible matches).
        
        Args:
            image_path: Path to the query image (optional if image_data provided)
            image_data: Raw image data as bytes (optional if image_path provided)
            threshold: Similarity threshold for matching
            
        Returns:
            Dictionary with identification results
        """
        try:
            # Load image from path or data
            if image_path:
                image = self.face_detector.load_image(image_path)
            elif image_data:
                image = self._decode_image_data(image_data)
            else:
                return {
                    'success': False,
                    'error': 'Either image_path or image_data must be provided',
                    'matches': []
                }
            
            if image is None:
                return {
                    'success': False,
                    'error': 'Could not load image',
                    'matches': []
                }
            
            # Detect and extract faces
            faces = self.face_detector.extract_faces_from_image(image)
            if not faces:
                return {
                    'success': False,
                    'error': 'No faces detected in image',
                    'matches': []
                }
            
            # Use the first detected face for identification
            face = faces[0]
            
            # Extract embedding
            embedding = self.face_embedder.extract_embedding(face)
            if embedding is None:
                return {
                    'success': False,
                    'error': 'Could not extract face embedding',
                    'matches': []
                }
            
            # Encrypt query embedding
            encrypted_data, iv = self.encryptor.encrypt_embedding(embedding)
            
            # Search for similar embeddings
            threshold = threshold or config.SIMILARITY_THRESHOLD
            similar_embeddings = self.storage.search_similar(
                query_encrypted_embedding=encrypted_data,
                n_results=10,
                threshold=threshold
            )
            
            return {
                'success': True,
                'matches': similar_embeddings,
                'total_matches': len(similar_embeddings)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Identification failed: {str(e)}',
                'matches': []
            }
    
    def compare_faces(self, image1_path: str = None, image2_path: str = None, 
                     image1_data: bytes = None, image2_data: bytes = None) -> Dict:
        """
        Compare two face images directly.
        
        Args:
            image1_path: Path to first image (optional if image1_data provided)
            image2_path: Path to second image (optional if image2_data provided)
            image1_data: Raw data for first image (optional if image1_path provided)
            image2_data: Raw data for second image (optional if image2_path provided)
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Load images
            if image1_path:
                image1 = self.face_detector.load_image(image1_path)
            elif image1_data:
                image1 = self._decode_image_data(image1_data)
            else:
                image1 = None
                
            if image2_path:
                image2 = self.face_detector.load_image(image2_path)
            elif image2_data:
                image2 = self._decode_image_data(image2_data)
            else:
                image2 = None
            
            if image1 is None or image2 is None:
                return {
                    'success': False,
                    'error': 'Could not load one or both images',
                    'similarity': 0.0
                }
            
            # Extract faces
            faces1 = self.face_detector.extract_faces_from_image(image1)
            faces2 = self.face_detector.extract_faces_from_image(image2)
            
            if not faces1 or not faces2:
                return {
                    'success': False,
                    'error': 'No faces detected in one or both images',
                    'similarity': 0.0
                }
            
            # Extract embeddings
            embedding1 = self.face_embedder.extract_embedding(faces1[0])
            embedding2 = self.face_embedder.extract_embedding(faces2[0])
            
            if embedding1 is None or embedding2 is None:
                return {
                    'success': False,
                    'error': 'Could not extract embeddings from one or both faces',
                    'similarity': 0.0
                }
            
            # Compute similarity
            similarity = self.face_embedder.compute_similarity(embedding1, embedding2)
            
            return {
                'success': True,
                'similarity': similarity,
                'same_person': similarity >= config.SIMILARITY_THRESHOLD
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Face comparison failed: {str(e)}',
                'similarity': 0.0
            }
    
    def get_user_info(self, user_id: str) -> Dict:
        """
        Get information about a user's stored embeddings.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with user information
        """
        try:
            embeddings = self.storage.get_embeddings_by_user(user_id)
            
            return {
                'success': True,
                'user_id': user_id,
                'total_embeddings': len(embeddings),
                'embeddings': embeddings
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Could not get user info: {str(e)}',
                'user_id': user_id
            }
    
    def delete_user(self, user_id: str) -> Dict:
        """
        Delete all embeddings for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with deletion results
        """
        try:
            deleted_count = self.storage.delete_user_embeddings(user_id)
            
            return {
                'success': True,
                'user_id': user_id,
                'deleted_embeddings': deleted_count
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Could not delete user: {str(e)}',
                'user_id': user_id
            }
    
    def get_system_stats(self) -> Dict:
        """
        Get system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        try:
            stats = self.storage.get_collection_stats()
            return {
                'success': True,
                'total_embeddings': stats['total_embeddings'],
                'collection_name': stats['collection_name']
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Could not get system stats: {str(e)}'
            }
    
    def _decode_image_data(self, image_data: bytes) -> Optional[np.ndarray]:
        """
        Decode image data from bytes.
        
        Args:
            image_data: Raw image data as bytes
            
        Returns:
            Image as numpy array or None if decoding fails
        """
        try:
            # Try to decode as base64 first
            if isinstance(image_data, str):
                # Remove data URL prefix if present
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_data = base64.b64decode(image_data)
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return image
        except Exception as e:
            print(f"Error decoding image data: {e}")
            return None
    
    def _save_user_image(self, user_id: str, image: np.ndarray, operation: str) -> Optional[str]:
        """
        Save user image to data directory.
        
        Args:
            user_id: User identifier
            image: Image as numpy array
            operation: Operation type (enrollment, authentication, etc.)
            
        Returns:
            Path to saved image or None if saving fails
        """
        try:
            import datetime
            
            # Create user directory
            user_dir = config.USER_IMAGES_DIR / user_id
            user_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{operation}_{timestamp}.jpg"
            image_path = user_dir / filename
            
            # Save image
            success = cv2.imwrite(str(image_path), image)
            
            if success:
                print(f"Image saved: {image_path}")
                return str(image_path)
            else:
                print(f"Failed to save image: {image_path}")
                return None
                
        except Exception as e:
            print(f"Error saving image: {e}")
            return None
    
    def _save_temp_image(self, image: np.ndarray, prefix: str = "temp") -> Optional[str]:
        """
        Save temporary image for processing.
        
        Args:
            image: Image as numpy array
            prefix: Filename prefix
            
        Returns:
            Path to saved temporary image
        """
        try:
            import datetime
            import uuid
            
            # Generate unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{prefix}_{timestamp}_{unique_id}.jpg"
            temp_path = config.TEMP_DIR / filename
            
            # Save image
            success = cv2.imwrite(str(temp_path), image)
            
            if success:
                return str(temp_path)
            else:
                return None
                
        except Exception as e:
            print(f"Error saving temporary image: {e}")
            return None
    
    def get_user_images(self, user_id: str) -> List[str]:
        """
        Get all saved images for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of image paths
        """
        try:
            user_dir = config.USER_IMAGES_DIR / user_id
            if user_dir.exists():
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                    image_files.extend(user_dir.glob(ext))
                return [str(path) for path in sorted(image_files)]
            return []
        except Exception as e:
            print(f"Error getting user images: {e}")
            return []
    
    def cleanup_temp_images(self, max_age_hours: int = 24):
        """
        Clean up old temporary images.
        
        Args:
            max_age_hours: Maximum age of temp images in hours
        """
        try:
            import datetime
            import time
            
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for temp_file in config.TEMP_DIR.glob("*.jpg"):
                file_age = current_time - temp_file.stat().st_mtime
                if file_age > max_age_seconds:
                    temp_file.unlink()
                    print(f"Cleaned up old temp file: {temp_file}")
                    
        except Exception as e:
            print(f"Error cleaning up temp images: {e}")