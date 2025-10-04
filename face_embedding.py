"""
Face embedding extraction using FaceNet.
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from typing import List, Optional
import config


class FaceEmbedder:
    """Face embedding extraction using pre-trained FaceNet model."""
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize the face embedder.
        
        Args:
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load pre-trained FaceNet model
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # MTCNN for face detection and alignment
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=self.device
        )
    
    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for FaceNet.
        
        Args:
            face_image: Face image as numpy array (BGR format)
            
        Returns:
            Preprocessed tensor ready for FaceNet
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Apply MTCNN preprocessing (normalization, etc.)
        processed_tensor = self.mtcnn(pil_image)
        
        if processed_tensor is not None:
            # Add batch dimension
            processed_tensor = processed_tensor.unsqueeze(0)
            return processed_tensor.to(self.device)
        else:
            return None
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from face image.
        
        Args:
            face_image: Face image as numpy array (BGR format)
            
        Returns:
            Face embedding as numpy array or None if extraction fails
        """
        try:
            # Preprocess the face
            processed_tensor = self.preprocess_face(face_image)
            
            if processed_tensor is None:
                return None
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(processed_tensor)
                # L2 normalize the embedding
                embedding = F.normalize(embedding, p=2, dim=1)
            
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def extract_embeddings_batch(self, face_images: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings from multiple face images.
        
        Args:
            face_images: List of face images as numpy arrays
            
        Returns:
            List of embeddings (None for failed extractions)
        """
        embeddings = []
        
        for face_image in face_images:
            embedding = self.extract_embedding(face_image)
            embeddings.append(embedding)
        
        return embeddings
    
    def detect_and_extract_embedding(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect faces in image and extract embeddings for each face.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of face embeddings
        """
        # Convert BGR to RGB for MTCNN
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect and crop faces
        face_tensors = self.mtcnn(rgb_image)
        
        embeddings = []
        
        if face_tensors is not None:
            # Handle single face
            if face_tensors.dim() == 3:
                face_tensors = face_tensors.unsqueeze(0)
            
            # Extract embeddings for all detected faces
            with torch.no_grad():
                face_embeddings = self.model(face_tensors.to(self.device))
                # L2 normalize
                face_embeddings = F.normalize(face_embeddings, p=2, dim=1)
                
                for embedding in face_embeddings:
                    embeddings.append(embedding.cpu().numpy().flatten())
        
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Ensure embeddings are normalized
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)
