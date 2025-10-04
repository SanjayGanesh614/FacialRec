"""
Matching module for face recognition and authentication.
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from face_detection import FaceDetector
from face_embedding import FaceEmbedder
from encryption import EmbeddingEncryptor
from chromadb_storage import ChromaDBStorage
import config


class FaceMatcher:
    """Face matching and authentication module."""
    
    def __init__(self, password: Optional[str] = None, device: str = 'auto'):
        """
        Initialize the face matcher.
        
        Args:
            password: Password for encryption key derivation
            device: Device to run models on ('cpu', 'cuda', or 'auto')
        """
        self.face_detector = FaceDetector(device=device)
        self.face_embedder = FaceEmbedder(device=device)
        self.encryptor = EmbeddingEncryptor(password=password)
        self.storage = ChromaDBStorage()
    
    def enroll_user(self, user_id: str, image_path: str, metadata: Dict = None) -> Dict:
        """
        Enroll a new user by processing their face image.
        
        Args:
            user_id: Unique identifier for the user
            image_path: Path to the user's face image
            metadata: Additional metadata to store
            
        Returns:
            Dictionary with enrollment results
        """
        try:
            # Load image
            image = self.face_detector.load_image(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': 'Could not load image',
                    'user_id': user_id
                }
            
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
                'embedding_ids': embedding_ids
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Enrollment failed: {str(e)}',
                'user_id': user_id
            }
    
    def authenticate_user(self, image_path: str, threshold: float = None) -> Dict:
        """
        Authenticate a user by comparing their face with stored embeddings.
        
        Args:
            image_path: Path to the query image
            threshold: Similarity threshold for matching (uses config default if None)
            
        Returns:
            Dictionary with authentication results
        """
        try:
            # Load image
            image = self.face_detector.load_image(image_path)
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
    
    def identify_user(self, image_path: str, threshold: float = None) -> Dict:
        """
        Identify a user from their face (returns all possible matches).
        
        Args:
            image_path: Path to the query image
            threshold: Similarity threshold for matching
            
        Returns:
            Dictionary with identification results
        """
        try:
            # Load image
            image = self.face_detector.load_image(image_path)
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
    
    def compare_faces(self, image1_path: str, image2_path: str) -> Dict:
        """
        Compare two face images directly.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Load images
            image1 = self.face_detector.load_image(image1_path)
            image2 = self.face_detector.load_image(image2_path)
            
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
