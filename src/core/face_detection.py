"""
Face detection module using OpenCV Haar Cascades.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from . import config
import os


class FaceDetector:
    """Face detection using OpenCV Haar Cascades and DNN."""
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize the face detector.
        
        Args:
            device: Device to run the model on (not used with OpenCV)
        """
        # Load multiple cascade classifiers for better detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Try to load DNN face detector for better accuracy
        try:
            # Download and use OpenCV's DNN face detector
            self.net = cv2.dnn.readNetFromTensorflow(
                model=None,  # Will use built-in model
                config=None
            )
            self.use_dnn = False  # Set to False initially, can be enabled if model is available
        except:
            self.net = None
            self.use_dnn = False
        
        if self.face_cascade.empty():
            print("Warning: Could not load face cascade classifier")
        
        print(f"Face detector initialized with {'DNN + ' if self.use_dnn else ''}Haar cascades")
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in an image using multiple methods for better accuracy.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of face detection results with bounding boxes
        """
        faces = []
        
        # Convert to grayscale for Haar cascade detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Frontal face detection
        face_rects = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More sensitive
            minNeighbors=3,    # Less strict
            minSize=(20, 20),  # Smaller minimum size
            maxSize=(300, 300), # Maximum size limit
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in face_rects:
            faces.append({
                'box': [x, y, w, h],
                'confidence': 0.8,  # High confidence for frontal faces
                'method': 'frontal'
            })
        
        # Method 2: Profile face detection (if no frontal faces found)
        if len(faces) == 0:
            profile_rects = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20),
                maxSize=(300, 300),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in profile_rects:
                faces.append({
                    'box': [x, y, w, h],
                    'confidence': 0.6,  # Lower confidence for profile faces
                    'method': 'profile'
                })
        
        # Method 3: Try different parameters if still no faces
        if len(faces) == 0:
            # More aggressive detection
            face_rects = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.02,
                minNeighbors=2,
                minSize=(15, 15),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in face_rects:
                faces.append({
                    'box': [x, y, w, h],
                    'confidence': 0.5,  # Lower confidence for aggressive detection
                    'method': 'aggressive'
                })
        
        # Remove overlapping detections
        faces = self._remove_overlapping_faces(faces)
        
        return faces
    
    def _remove_overlapping_faces(self, faces: List[dict]) -> List[dict]:
        """Remove overlapping face detections."""
        if len(faces) <= 1:
            return faces
        
        # Sort by confidence
        faces.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered_faces = []
        for face in faces:
            x1, y1, w1, h1 = face['box']
            
            # Check if this face overlaps significantly with any already added face
            overlap = False
            for existing_face in filtered_faces:
                x2, y2, w2, h2 = existing_face['box']
                
                # Calculate intersection over union (IoU)
                intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * \
                                  max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                union_area = w1 * h1 + w2 * h2 - intersection_area
                
                if union_area > 0:
                    iou = intersection_area / union_area
                    if iou > 0.3:  # 30% overlap threshold
                        overlap = True
                        break
            
            if not overlap:
                filtered_faces.append(face)
        
        return filtered_faces
    
    def extract_face(self, image: np.ndarray, face_info: dict) -> Optional[np.ndarray]:
        """
        Extract and align face from image using face detection results.
        
        Args:
            image: Input image as numpy array (BGR format)
            face_info: Face detection result
            
        Returns:
            Cropped and aligned face image or None if extraction fails
        """
        try:
            # Get bounding box
            x, y, w, h = face_info['box']
            
            # Add margin
            margin = config.MARGIN
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2 * margin)
            h = min(image.shape[0] - y, h + 2 * margin)
            
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
    
    def capture_from_array(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        """
        Process image from numpy array (for camera capture).
        
        Args:
            image_array: Image as numpy array
            
        Returns:
            Image as numpy array or None if processing fails
        """
        try:
            if image_array is None:
                return None
            return image_array
        except Exception as e:
            print(f"Error processing image array: {e}")
            return None