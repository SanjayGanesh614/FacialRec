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
        except:
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
        # We'll store the encrypted data as a list of bytes converted to floats
        encrypted_list = list(encrypted_embedding)
        
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
            embeddings_list.append(list(encrypted_embedding))
        
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
        Search for similar encrypted embeddings.
        
        Args:
            query_encrypted_embedding: Encrypted query embedding
            n_results: Number of results to return
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar embeddings with metadata and distances
        """
        # Convert query to list
        query_list = list(query_encrypted_embedding)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=n_results,
            include=['metadatas', 'distances']
        )
        
        # Process results
        similar_embeddings = []
        
        if results['ids'] and results['ids'][0]:
            for i, (embedding_id, distance, metadata) in enumerate(zip(
                results['ids'][0],
                results['distances'][0],
                results['metadatas'][0]
            )):
                # Convert distance to similarity (ChromaDB uses cosine distance)
                similarity = 1 - distance
                
                # Apply threshold if specified
                if threshold is not None and similarity < threshold:
                    continue
                
                similar_embeddings.append({
                    'id': embedding_id,
                    'user_id': metadata['user_id'],
                    'similarity': similarity,
                    'distance': distance,
                    'metadata': metadata
                })
        
        return similar_embeddings
    
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
                    'embedding': bytes(results['embeddings'][0]),
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
                    'embedding': bytes(results['embeddings'][i]),
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
