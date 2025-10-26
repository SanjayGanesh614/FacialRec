"""
ChromaDB storage module for encrypted face embeddings.
"""
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional, Tuple
import uuid
from . import config


class ChromaDBStorage:
    """ChromaDB storage for encrypted face embeddings."""
    
    def __init__(self, persist_directory: str = None):
        """
        Initialize ChromaDB storage.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory or config.CHROMA_PERSIST_DIRECTORY
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(config.CHROMA_COLLECTION_NAME)
            print(f"Using existing ChromaDB collection: {config.CHROMA_COLLECTION_NAME}")
        except Exception as e:
            print(f"Creating new ChromaDB collection: {config.CHROMA_COLLECTION_NAME}")
            self.collection = self.client.create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={"description": "Encrypted face embeddings storage"}
            )
    
    def add_embedding(self, user_id: str, encrypted_embedding: bytes, metadata: Dict = None) -> str:
        """
        Add an encrypted embedding to the database.
        
        Args:
            user_id: Unique identifier for the user
            encrypted_embedding: Encrypted face embedding as bytes
            metadata: Additional metadata to store
            
        Returns:
            Unique ID of the stored embedding
        """
        # Generate unique ID for this embedding
        embedding_id = str(uuid.uuid4())
        
        # Prepare metadata
        embedding_metadata = {
            "user_id": user_id,
            "encrypted": True,
            "embedding_dim": len(encrypted_embedding)
        }
        
        if metadata:
            embedding_metadata.update(metadata)
        
        # Convert encrypted bytes to list for ChromaDB
        # ChromaDB expects embeddings as lists of floats
        # We'll store the encrypted data as a list of bytes converted to floats (0-255 range)
        encrypted_list = [float(b) for b in encrypted_embedding]
        
        # Add to collection
        self.collection.add(
            ids=[embedding_id],
            embeddings=[encrypted_list],  # ChromaDB will handle the vector operations
            metadatas=[embedding_metadata]
        )
        
        return embedding_id
    
    def add_embeddings_batch(self, user_ids: List[str], encrypted_embeddings: List[bytes], 
                            metadatas: List[Dict] = None) -> List[str]:
        """
        Add multiple encrypted embeddings to the database.
        
        Args:
            user_ids: List of user identifiers
            encrypted_embeddings: List of encrypted face embeddings
            metadatas: List of metadata dictionaries
            
        Returns:
            List of unique IDs for stored embeddings
        """
        embedding_ids = []
        embeddings_list = []
        metadata_list = []
        
        for i, (user_id, encrypted_embedding) in enumerate(zip(user_ids, encrypted_embeddings)):
            embedding_id = str(uuid.uuid4())
            embedding_ids.append(embedding_id)
            
            # Prepare metadata
            embedding_metadata = {
                "user_id": user_id,
                "encrypted": True,
                "embedding_dim": len(encrypted_embedding)
            }
            
            if metadatas and i < len(metadatas):
                embedding_metadata.update(metadatas[i])
            
            metadata_list.append(embedding_metadata)
            
            # Convert to list
            embeddings_list.append([float(b) for b in encrypted_embedding])
        
        # Add to collection
        self.collection.add(
            ids=embedding_ids,
            embeddings=embeddings_list,
            metadatas=metadata_list
        )
        
        return embedding_ids
    
    def search_similar(self, query_encrypted_embedding: bytes, n_results: int = 5, 
                      threshold: float = None) -> List[Dict]:
        """
        Search for similar encrypted embeddings by decrypting and comparing.
        
        Args:
            query_encrypted_embedding: Encrypted query embedding
            n_results: Number of results to return
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar embeddings with metadata and distances
        """
        try:
            # Get all embeddings from the database
            try:
                all_results = self.collection.get(
                    include=['embeddings', 'metadatas']
                )
            except Exception as e:
                print(f"Error getting embeddings from collection: {e}")
                return []
            
            if not all_results['ids']:
                return []
            
            # We need to decrypt and compare embeddings manually
            # This is a temporary solution - in production, you'd want a more efficient approach
            from .encryption import EmbeddingEncryptor
            encryptor = EmbeddingEncryptor()
            
            # Decrypt query embedding
            query_iv = query_encrypted_embedding[:16]  # First 16 bytes as IV
            query_encrypted_data = query_encrypted_embedding[16:]  # Rest is encrypted data
            query_embedding = encryptor.decrypt_embedding(query_encrypted_data, query_iv)
            
            if query_embedding is None:
                print("Failed to decrypt query embedding")
                return []
            
            similarities = []
            
            # Compare with all stored embeddings
            print(f"Comparing query with {len(all_results['ids'])} stored embeddings")
            
            for i, (embedding_id, encrypted_data, metadata) in enumerate(zip(
                all_results['ids'],
                all_results['embeddings'], 
                all_results['metadatas']
            )):
                try:
                    # Convert list back to bytes (handle float to int conversion)
                    encrypted_bytes = bytes([int(b) for b in encrypted_data])
                    
                    # Extract IV and encrypted data
                    stored_iv = encrypted_bytes[:16]  # First 16 bytes as IV
                    stored_encrypted_data = encrypted_bytes[16:]  # Rest is encrypted data
                    
                    # Decrypt stored embedding
                    stored_embedding = encryptor.decrypt_embedding(stored_encrypted_data, stored_iv)
                    
                    if stored_embedding is not None:
                        # Calculate cosine similarity
                        similarity = self._calculate_cosine_similarity(query_embedding, stored_embedding)
                        print(f"Similarity with {metadata['user_id']}: {similarity:.4f}")
                        
                        similarities.append({
                            'id': embedding_id,
                            'user_id': metadata['user_id'],
                            'similarity': float(similarity),
                            'distance': float(1 - similarity),
                            'metadata': metadata
                        })
                    else:
                        print(f"Failed to decrypt embedding {embedding_id}")
                        
                except Exception as e:
                    print(f"Error processing embedding {embedding_id}: {e}")
                    continue
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Apply threshold filter
            if threshold is not None:
                similarities = [s for s in similarities if s['similarity'] >= threshold]
            
            # Return top n_results
            return similarities[:n_results]
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    def _calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            print(f"Calculating similarity between embeddings of shapes: {embedding1.shape} and {embedding2.shape}")
            
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            print(f"Embedding norms: {norm1:.4f}, {norm2:.4f}")
            
            if norm1 == 0 or norm2 == 0:
                print("One of the embeddings has zero norm")
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            print(f"Raw cosine similarity: {similarity:.4f}")
            
            # Ensure result is in [0, 1] range
            similarity = (similarity + 1) / 2
            print(f"Normalized similarity: {similarity:.4f}")
            
            return float(similarity)
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_embedding_by_id(self, embedding_id: str) -> Optional[Dict]:
        """
        Get embedding by ID.
        
        Args:
            embedding_id: Unique embedding ID
            
        Returns:
            Embedding data with metadata or None if not found
        """
        try:
            results = self.collection.get(
                ids=[embedding_id],
                include=['embeddings', 'metadatas']
            )
            
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'embedding': bytes([int(b) for b in results['embeddings'][0]]),
                    'metadata': results['metadatas'][0]
                }
        except Exception as e:
            print(f"Error getting embedding by ID: {e}")
        
        return None
    
    def get_embeddings_by_user(self, user_id: str) -> List[Dict]:
        """
        Get all embeddings for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of embeddings for the user
        """
        try:
            results = self.collection.get(
                where={"user_id": user_id},
                include=['embeddings', 'metadatas']
            )
            
            embeddings = []
            for i, embedding_id in enumerate(results['ids']):
                embeddings.append({
                    'id': embedding_id,
                    'embedding': bytes([int(b) for b in results['embeddings'][i]]),
                    'metadata': results['metadatas'][i]
                })
            
            return embeddings
        except Exception as e:
            print(f"Error getting embeddings by user: {e}")
            return []
    
    def delete_embedding(self, embedding_id: str) -> bool:
        """
        Delete an embedding by ID.
        
        Args:
            embedding_id: Unique embedding ID
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            self.collection.delete(ids=[embedding_id])
            return True
        except Exception as e:
            print(f"Error deleting embedding: {e}")
            return False
    
    def delete_user_embeddings(self, user_id: str) -> int:
        """
        Delete all embeddings for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Number of embeddings deleted
        """
        try:
            # Get all embeddings for user
            user_embeddings = self.get_embeddings_by_user(user_id)
            
            if not user_embeddings:
                return 0
            
            # Delete all embeddings
            embedding_ids = [emb['id'] for emb in user_embeddings]
            self.collection.delete(ids=embedding_ids)
            
            return len(embedding_ids)
        except Exception as e:
            print(f"Error deleting user embeddings: {e}")
            return 0
    
    def get_collection_stats(self) -> Dict:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'total_embeddings': count,
                'collection_name': config.CHROMA_COLLECTION_NAME
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}")
            return {'total_embeddings': 0, 'collection_name': config.CHROMA_COLLECTION_NAME}
    
    def reset_collection(self):
        """Reset the entire collection (delete all data)."""
        try:
            self.client.delete_collection(config.CHROMA_COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={"description": "Encrypted face embeddings storage"}
            )
            print("Collection reset successfully")
        except Exception as e:
            print(f"Error resetting collection: {e}")
