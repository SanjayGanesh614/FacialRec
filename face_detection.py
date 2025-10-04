"""
Face detection module using MTCNN.
"""
import cv2
import numpy as np
from PIL import Image
import torch
from mtcnn import MTCNN
from typing import List, Tuple, Optional
import config


class FaceDetector:
    """Face detection using MTCNN."""
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize the face detector.
        
        Args:
            device: Device to run the model on ('cpu', 'cuda', or 'auto')
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.detector = MTCNN(
            device=self.device,
            min_face_size=config.MIN_FACE_SIZE,
            margin=config.MARGIN
        )
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of face detection results with bounding boxes and landmarks
        """
        # Convert BGR to RGB for MTCNN
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.detector.detect_faces(rgb_image)
        
        return faces
    
    def extract_face(self, image: np.ndarray, face_info: dict) -> Optional[np.ndarray]:
        """
        Extract and align face from image using face detection results.
        
        Args:
            image: Input image as numpy array (BGR format)
            face_info: Face detection result from MTCNN
            
        Returns:
            Cropped and aligned face image or None if extraction fails
        """
        try:
            # Get bounding box
            x, y, w, h = face_info['box']
            
            # Add margin
            x = max(0, x - config.MARGIN)
            y = max(0, y - config.MARGIN)
            w = min(image.shape[1] - x, w + 2 * config.MARGIN)
            h = min(image.shape[0] - y, h + 2 * config.MARGIN)
            
            # Extract face region
            face_crop = image[y:y+h, x:x+w]
            
            # Resize to required size
            face_resized = cv2.resize(face_crop, config.FACE_SIZE)
            
            return face_resized
            
        except Exception as e:
            print(f"Error extracting face: {e}")
            return None
    
    def extract_faces_from_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract all faces from an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of cropped and aligned face images
        """
        faces = self.detect_faces(image)
        extracted_faces = []
        
        for face_info in faces:
            face = self.extract_face(image, face_info)
            if face is not None:
                extracted_faces.append(face)
        
        return extracted_faces
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array or None if loading fails
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image: {image_path}")
                return None
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
