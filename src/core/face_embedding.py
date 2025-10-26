"""
Face embedding extraction using OpenCV and simple feature extraction.
"""
import numpy as np
import cv2
from typing import List, Optional
from sklearn.preprocessing import normalize
from . import config


class FaceEmbedder:
    """Face embedding extraction using OpenCV features."""
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize the face embedder.
        
        Args:
            device: Device to run the model on (not used with OpenCV)
        """
        print("Initializing OpenCV-based face embedder...")
        # Initialize feature extractors
        self.orb = cv2.ORB_create(nfeatures=500)
        self.sift = cv2.SIFT_create(nfeatures=100)
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from face image using advanced feature descriptors.
        
        Args:
            face_image: Face image as numpy array (BGR format)
            
        Returns:
            Face embedding as numpy array or None if extraction fails
        """
        try:
            # Convert to grayscale and resize
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, config.FACE_SIZE)
            
            # Enhance image quality
            gray = cv2.equalizeHist(gray)  # Histogram equalization
            gray = cv2.GaussianBlur(gray, (3, 3), 0)  # Slight blur to reduce noise
            
            features = []
            
            # 1. Enhanced Histogram Features (multiple regions)
            hist_features = self._extract_regional_histograms(gray)
            features.extend(hist_features)
            
            # 2. Local Binary Pattern (LBP) Features
            lbp_features = self._extract_lbp_features(gray)
            features.extend(lbp_features)
            
            # 3. Gradient and Edge Features
            gradient_features = self._extract_gradient_features(gray)
            features.extend(gradient_features)
            
            # 4. Texture Features (GLCM-inspired)
            texture_features = self._extract_texture_features(gray)
            features.extend(texture_features)
            
            # 5. Geometric Features
            geometric_features = self._extract_geometric_features(gray)
            features.extend(geometric_features)
            
            # 6. Frequency Domain Features
            frequency_features = self._extract_frequency_features(gray)
            features.extend(frequency_features)
            
            # Convert to numpy array and normalize
            embedding = np.array(features, dtype=np.float32)
            
            # Remove any NaN or infinite values
            embedding = np.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # L2 normalize
            embedding = normalize([embedding])[0]
            
            # Ensure fixed size
            target_size = config.EMBEDDING_DIM
            if len(embedding) > target_size:
                embedding = embedding[:target_size]
            elif len(embedding) < target_size:
                padding = np.zeros(target_size - len(embedding))
                embedding = np.concatenate([embedding, padding])
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def _extract_regional_histograms(self, gray_image: np.ndarray) -> List[float]:
        """Extract histogram features from different regions of the face."""
        features = []
        h, w = gray_image.shape
        
        # Divide face into regions (eyes, nose, mouth areas)
        regions = [
            gray_image[0:h//3, 0:w//3],      # Top-left (left eye area)
            gray_image[0:h//3, 2*w//3:w],    # Top-right (right eye area)
            gray_image[h//3:2*h//3, w//4:3*w//4],  # Center (nose area)
            gray_image[2*h//3:h, w//4:3*w//4],     # Bottom (mouth area)
            gray_image  # Full face
        ]
        
        for region in regions:
            if region.size > 0:
                hist = cv2.calcHist([region], [0], None, [16], [0, 256])
                hist = hist.flatten() / (hist.sum() + 1e-7)
                features.extend(hist)
        
        return features[:48]  # Limit to 48 features
    
    def _extract_gradient_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract gradient and edge features."""
        features = []
        
        # Sobel gradients
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_dir = np.arctan2(grad_y, grad_x)
        
        # Gradient magnitude histogram
        mag_hist = cv2.calcHist([grad_mag.astype(np.uint8)], [0], None, [16], [0, 256])
        mag_hist = mag_hist.flatten() / (mag_hist.sum() + 1e-7)
        features.extend(mag_hist)
        
        # Gradient direction histogram
        dir_hist, _ = np.histogram(grad_dir, bins=8, range=(-np.pi, np.pi))
        dir_hist = dir_hist.astype(np.float32) / (dir_hist.sum() + 1e-7)
        features.extend(dir_hist)
        
        # Canny edges
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        return features[:25]  # Limit to 25 features
    
    def _extract_texture_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract texture features using statistical measures."""
        features = []
        
        # Statistical measures
        features.append(np.mean(gray_image) / 255.0)
        features.append(np.std(gray_image) / 255.0)
        features.append(np.var(gray_image) / (255.0**2))
        
        # Entropy
        hist, _ = np.histogram(gray_image, bins=256, range=(0, 256))
        hist = hist.astype(np.float32) / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        features.append(entropy / 8.0)  # Normalize
        
        # Contrast and homogeneity (simplified GLCM)
        contrast = 0
        homogeneity = 0
        for i in range(gray_image.shape[0] - 1):
            for j in range(gray_image.shape[1] - 1):
                diff = abs(int(gray_image[i, j]) - int(gray_image[i+1, j+1]))
                contrast += diff**2
                homogeneity += 1.0 / (1.0 + diff)
        
        total_pairs = (gray_image.shape[0] - 1) * (gray_image.shape[1] - 1)
        features.append(contrast / (total_pairs * 255**2))
        features.append(homogeneity / total_pairs)
        
        return features[:6]  # Limit to 6 features
    
    def _extract_geometric_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract geometric features from face structure."""
        features = []
        
        # Face symmetry (compare left and right halves)
        h, w = gray_image.shape
        left_half = gray_image[:, :w//2]
        right_half = cv2.flip(gray_image[:, w//2:], 1)  # Flip right half
        
        # Resize to same size if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # Calculate symmetry score
        symmetry = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
        features.append(symmetry if not np.isnan(symmetry) else 0.0)
        
        # Aspect ratio
        features.append(h / w)
        
        # Center of mass
        moments = cv2.moments(gray_image)
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00'] / w
            cy = moments['m01'] / moments['m00'] / h
        else:
            cx, cy = 0.5, 0.5
        features.extend([cx, cy])
        
        return features[:4]  # Limit to 4 features
    
    def _extract_frequency_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract frequency domain features using DCT."""
        features = []
        
        # Discrete Cosine Transform
        dct = cv2.dct(np.float32(gray_image))
        
        # Take low-frequency components (top-left 8x8 block)
        dct_block = dct[:8, :8]
        dct_features = dct_block.flatten()
        
        # Normalize and take most significant coefficients
        dct_features = dct_features / (np.max(np.abs(dct_features)) + 1e-7)
        features.extend(dct_features[:16])  # Take first 16 coefficients
        
        return features[:16]  # Limit to 16 features
    
    def _extract_lbp_features(self, gray_image: np.ndarray) -> List[float]:
        """Extract simplified Local Binary Pattern features."""
        try:
            h, w = gray_image.shape
            lbp_features = []
            
            # Sample points in a grid
            step = 8
            for y in range(step, h-step, step):
                for x in range(step, w-step, step):
                    center = gray_image[y, x]
                    
                    # Compare with 8 neighbors
                    neighbors = [
                        gray_image[y-1, x-1], gray_image[y-1, x], gray_image[y-1, x+1],
                        gray_image[y, x+1], gray_image[y+1, x+1], gray_image[y+1, x],
                        gray_image[y+1, x-1], gray_image[y, x-1]
                    ]
                    
                    # Create binary pattern
                    binary_pattern = 0
                    for i, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            binary_pattern |= (1 << i)
                    
                    lbp_features.append(binary_pattern / 255.0)
            
            return lbp_features[:64]  # Limit to 64 features
            
        except Exception as e:
            print(f"Error extracting LBP features: {e}")
            return [0.0] * 64
    
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
        try:
            from .face_detection import FaceDetector
            
            detector = FaceDetector()
            faces = detector.extract_faces_from_image(image)
            
            embeddings = []
            for face in faces:
                embedding = self.extract_embedding(face)
                if embedding is not None:
                    embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            print(f"Error detecting and extracting embeddings: {e}")
            return []
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1, where 1 is most similar)
        """
        try:
            # Ensure embeddings are normalized
            embedding1 = normalize([embedding1])[0]
            embedding2 = normalize([embedding2])[0]
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2)
            
            # Ensure result is in [0, 1] range
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0