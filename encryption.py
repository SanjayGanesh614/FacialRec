"""
AES encryption module for face embeddings.
"""
import os
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import numpy as np
from typing import Optional, Tuple
import config


class EmbeddingEncryptor:
    """AES encryption for face embeddings."""
    
    def __init__(self, password: Optional[str] = None):
        """
        Initialize the encryptor.
        
        Args:
            password: Password for key derivation. If None, will try to load from file.
        """
        self.key = self._get_or_create_key(password)
    
    def _get_or_create_key(self, password: Optional[str] = None) -> bytes:
        """
        Get existing key or create new one.
        
        Args:
            password: Password for key derivation
            
        Returns:
            AES key as bytes
        """
        if password is not None:
            # Derive key from password
            return self._derive_key_from_password(password)
        
        # Try to load existing key
        if config.KEY_FILE_PATH.exists():
            try:
                with open(config.KEY_FILE_PATH, 'rb') as f:
                    return f.read()
            except Exception as e:
                print(f"Error loading key file: {e}")
        
        # Generate new key
        key = os.urandom(config.AES_KEY_SIZE)
        
        # Save key to file
        try:
            with open(config.KEY_FILE_PATH, 'wb') as f:
                f.write(key)
            print(f"New encryption key saved to {config.KEY_FILE_PATH}")
        except Exception as e:
            print(f"Warning: Could not save key to file: {e}")
        
        return key
    
    def _derive_key_from_password(self, password: str) -> bytes:
        """
        Derive AES key from password using PBKDF2.
        
        Args:
            password: Password string
            
        Returns:
            Derived AES key
        """
        # Generate salt (you might want to store this securely)
        salt = b'face_recognition_salt_2024'  # In production, use random salt per user
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=config.AES_KEY_SIZE,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        return kdf.derive(password.encode())
    
    def _generate_iv(self) -> bytes:
        """Generate random IV for encryption."""
        return os.urandom(config.AES_IV_SIZE)
    
    def encrypt_embedding(self, embedding: np.ndarray) -> Tuple[bytes, bytes]:
        """
        Encrypt a face embedding.
        
        Args:
            embedding: Face embedding as numpy array
            
        Returns:
            Tuple of (encrypted_data, iv)
        """
        try:
            # Convert embedding to bytes
            embedding_bytes = embedding.astype(np.float32).tobytes()
            
            # Generate random IV
            iv = self._generate_iv()
            
            # Create cipher
            cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            # Pad the data to block size
            padding_length = 16 - (len(embedding_bytes) % 16)
            padded_data = embedding_bytes + bytes([padding_length] * padding_length)
            
            # Encrypt
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            return encrypted_data, iv
            
        except Exception as e:
            print(f"Error encrypting embedding: {e}")
            raise
    
    def decrypt_embedding(self, encrypted_data: bytes, iv: bytes) -> Optional[np.ndarray]:
        """
        Decrypt a face embedding.
        
        Args:
            encrypted_data: Encrypted embedding data
            iv: Initialization vector
            
        Returns:
            Decrypted embedding as numpy array or None if decryption fails
        """
        try:
            # Create cipher
            cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()
            
            # Decrypt
            decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()
            
            # Remove padding
            padding_length = decrypted_padded[-1]
            decrypted_data = decrypted_padded[:-padding_length]
            
            # Convert back to numpy array
            embedding = np.frombuffer(decrypted_data, dtype=np.float32)
            
            return embedding
            
        except Exception as e:
            print(f"Error decrypting embedding: {e}")
            return None
    
    def encrypt_embedding_to_string(self, embedding: np.ndarray) -> str:
        """
        Encrypt embedding and return as base64 string.
        
        Args:
            embedding: Face embedding as numpy array
            
        Returns:
            Base64 encoded string containing encrypted data and IV
        """
        encrypted_data, iv = self.encrypt_embedding(embedding)
        
        # Combine encrypted data and IV
        combined = iv + encrypted_data
        
        # Encode as base64
        return base64.b64encode(combined).decode('utf-8')
    
    def decrypt_embedding_from_string(self, encrypted_string: str) -> Optional[np.ndarray]:
        """
        Decrypt embedding from base64 string.
        
        Args:
            encrypted_string: Base64 encoded encrypted data
            
        Returns:
            Decrypted embedding as numpy array or None if decryption fails
        """
        try:
            # Decode base64
            combined = base64.b64decode(encrypted_string.encode('utf-8'))
            
            # Split IV and encrypted data
            iv = combined[:config.AES_IV_SIZE]
            encrypted_data = combined[config.AES_IV_SIZE:]
            
            # Decrypt
            return self.decrypt_embedding(encrypted_data, iv)
            
        except Exception as e:
            print(f"Error decrypting from string: {e}")
            return None
    
    def encrypt_embeddings_batch(self, embeddings: List[np.ndarray]) -> List[Tuple[bytes, bytes]]:
        """
        Encrypt multiple embeddings.
        
        Args:
            embeddings: List of face embeddings
            
        Returns:
            List of (encrypted_data, iv) tuples
        """
        encrypted_embeddings = []
        
        for embedding in embeddings:
            if embedding is not None:
                encrypted_data, iv = self.encrypt_embedding(embedding)
                encrypted_embeddings.append((encrypted_data, iv))
            else:
                encrypted_embeddings.append((None, None))
        
        return encrypted_embeddings
