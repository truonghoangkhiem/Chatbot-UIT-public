# DEPRECATED: This file is no longer used.
# 
# The system has been migrated to Weaviate vector database.
# Please use store/vector/weaviate_store.py instead.
#
# See WEAVIATE_MIGRATION.md for migration details.

# store/vector/chroma_store.py
#
# Description:
# This module provides a handler for interacting with a ChromaDB vector store.
# It manages the connection and collection, wrapping it in a LlamaIndex-compatible
# `ChromaVectorStore` object.
#
# Key Responsibilities:
# - Establishing a persistent connection to ChromaDB.
# - Getting or creating a specific collection for the RAG data.
# - Providing a singleton instance of the `ChromaVectorStore`.

from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Application-specific configurations
from app.config.settings import settings

# Global cache for the Chroma store instance
_chroma_store = None

def get_chroma_vector_store() -> ChromaVectorStore:
    """
    Factory function to get the singleton instance of the ChromaVectorStore.
    It initializes the client and collection on the first call and returns the
    cached instance on subsequent calls.

    Returns:
        ChromaVectorStore: The LlamaIndex-compatible ChromaDB vector store.
    """
    global _chroma_store
    if _chroma_store is not None:
        return _chroma_store

    # Initialize a persistent ChromaDB client
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    
    # Get or create the collection where vectors will be stored
    collection = client.get_or_create_collection("rag_collection")
    
    # Wrap the Chroma collection with LlamaIndex's vector store adapter
    _chroma_store = ChromaVectorStore(chroma_collection=collection)
    
    return _chroma_store