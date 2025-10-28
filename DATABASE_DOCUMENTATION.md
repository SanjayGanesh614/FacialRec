# Database Table Documentation: ChromaDB Collection for Face Embeddings

## Overview
This facial recognition system uses ChromaDB, a vector database, to store encrypted face embeddings. The "database table" is implemented as a ChromaDB collection, which serves as a vector store for similarity searches rather than a traditional relational table.

## Collection Setup

### Initialization
The collection is initialized through the `ChromaDBStorage` class in `src/core/chromadb_storage.py`. To set up the database:

1. **Import the storage class:**
   ```python
   from src.core.chromadb_storage import ChromaDBStorage
   ```

2. **Initialize the storage instance:**
   ```python
   storage = ChromaDBStorage(persist_directory="./chroma_db")
   ```
   - `persist_directory`: Directory where ChromaDB data will be persisted (defaults to `config.CHROMA_PERSIST_DIRECTORY`)

3. **Automatic Collection Creation:**
   - The class automatically attempts to get an existing collection named via `config.CHROMA_COLLECTION_NAME`
   - If the collection doesn't exist, it creates a new one with metadata: `{"description": "Encrypted face embeddings storage"}`

### Configuration
- **Collection Name:** Defined in `src/core/config.py` as `CHROMA_COLLECTION_NAME`
- **Persistence Directory:** Defined in `src/core/config.py` as `CHROMA_PERSIST_DIRECTORY`
- **Client Settings:** Uses `chromadb.PersistentClient` with anonymized telemetry disabled and reset allowed

## Table Structure

### Fields
Each "record" in the collection consists of three main components:

1. **ID (`ids`)**
   - Type: String (UUID)
   - Description: Unique identifier for each embedding
   - Generated: Automatically using `uuid.uuid4()` when adding embeddings

2. **Embedding (`embeddings`)**
   - Type: List of floats
   - Description: Encrypted face embedding data
   - Format: Encrypted bytes converted to float values (0-255 range)
   - Note: Original encrypted data includes IV (first 16 bytes) + encrypted embedding data

3. **Metadata (`metadatas`)**
   - Type: Dictionary
   - Required fields:
     - `user_id`: String - Unique identifier for the user
     - `encrypted`: Boolean - Always `True` (indicating data is encrypted)
     - `embedding_dim`: Integer - Length of the encrypted embedding in bytes
   - Optional fields: Any additional metadata can be added when storing embeddings

### Example Record Structure
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "embedding": [123.0, 45.0, 67.0, ...],  // Encrypted data as floats
  "metadata": {
    "user_id": "user123",
    "encrypted": true,
    "embedding_dim": 512,
    "timestamp": "2024-01-15T10:30:00Z",
    "device": "webcam"
  }
}
```

## Operations

### Adding Embeddings
- **Single Embedding:** Use `add_embedding(user_id, encrypted_embedding, metadata)`
- **Batch Embeddings:** Use `add_embeddings_batch(user_ids, encrypted_embeddings, metadatas)`

### Querying
- **Similarity Search:** `search_similar(query_encrypted_embedding, n_results, threshold)`
  - Decrypts query and stored embeddings for comparison
  - Returns sorted list by similarity score
- **Get by ID:** `get_embedding_by_id(embedding_id)`
- **Get by User:** `get_embeddings_by_user(user_id)`

### Maintenance
- **Delete Single:** `delete_embedding(embedding_id)`
- **Delete User Data:** `delete_user_embeddings(user_id)`
- **Reset Collection:** `reset_collection()` - Deletes all data and recreates collection
- **Statistics:** `get_collection_stats()` - Returns total embedding count

## Data Flow

1. **Face Detection:** MTCNN detects faces in images
2. **Embedding Generation:** FaceNet generates 512-dimensional embeddings
3. **Encryption:** Embeddings encrypted using AES with random IV
4. **Storage:** Encrypted data stored in ChromaDB collection
5. **Retrieval:** For matching, embeddings are decrypted and compared using cosine similarity

## Security Considerations

- All embeddings are stored encrypted
- IV (Initialization Vector) is prepended to encrypted data
- Decryption happens only during similarity comparisons
- No plain-text face data is persisted

## Performance Notes

- Similarity searches require decrypting all stored embeddings (not optimized for large datasets)
- For production use, consider more efficient vector databases like Pinecone or Weaviate
- Current implementation is suitable for small to medium-scale applications

## Dependencies

- `chromadb`: Vector database client
- `numpy`: For similarity calculations
- Custom modules: `config`, `encryption`

This setup provides a secure, vector-based storage solution for facial recognition embeddings with built-in encryption and similarity search capabilities.
