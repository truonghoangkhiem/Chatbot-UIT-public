# DEPRECATED: This file is no longer used.
# 
# The system has been migrated to Weaviate vector database.
# Please use store/vector/weaviate_store.py instead.
#
# See WEAVIATE_MIGRATION.md for migration details.

# store/vector/faiss_store.py
#
# Description:
# This module provides a handler for creating, loading, and persisting a Faiss index.
# Faiss is a library for efficient similarity search. This module wraps the Faiss index
# with LlamaIndex's `FaissVectorStore` for seamless integration.
#
# Key Responsibilities:
# - Loading a Faiss index from a file if it exists.
# - Creating a new Faiss index if one does not exist.
# - Providing a singleton instance of the `FaissVectorStore`.
# - Explicitly persisting the Faiss index to disk.

from pathlib import Path
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# Application-specific configurations
from app.config.settings import settings

# --- Module-level Constants and Caches ---
_INDEX_PATH = Path(settings.storage_dir) / "faiss.index"
_faiss_store = None

def get_faiss_vector_store() -> FaissVectorStore:
    """
    Factory function to create or load a Faiss index and wrap it in a
    LlamaIndex FaissVectorStore. Implements a singleton pattern.
    NOTE: The `index_path` argument is not passed to the constructor to ensure
          compatibility with the specific adapter version being used.

    Returns:
        FaissVectorStore: The LlamaIndex-compatible Faiss vector store.
    """
    global _faiss_store
    if _faiss_store is not None:
        return _faiss_store

    # Default embedding dimension for 'intfloat/multilingual-e5-base' is 768.
    # This must be updated if a different embedding model is used.
    dim = 768

    if _INDEX_PATH.exists():
        # Load the index from the file system
        index = faiss.read_index(str(_INDEX_PATH))
    else:
        # Create a new index. IndexFlatIP is suitable for inner product (IP) similarity,
        # which approximates cosine similarity for normalized embeddings.
        index = faiss.IndexFlatIP(dim)

    # Wrap the Faiss index with LlamaIndex's vector store adapter
    _faiss_store = FaissVectorStore(faiss_index=index)
    return _faiss_store

def persist_faiss(store: FaissVectorStore):
    """
    Explicitly persists the raw Faiss index to disk.
    This is necessary because the LlamaIndex wrapper may not handle persistence automatically.

    Args:
        store (FaissVectorStore): The vector store instance containing the index to persist.
    """
    # Access the internal Faiss index object
    idx = store._faiss_index  # type: ignore
    
    # Use the faiss library to write the index to the specified path
    faiss.write_index(idx, str(_INDEX_PATH))