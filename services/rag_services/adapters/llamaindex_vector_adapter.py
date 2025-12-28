# DEPRECATED: This adapter is no longer used.
# 
# The system has been migrated to Weaviate vector database.
# Please use adapters/weaviate_vector_adapter.py instead.
#
# See WEAVIATE_MIGRATION.md for migration details.

# adapters/llamaindex_vector_adapter.py
#
# Description:
# Adapter implementation for LlamaIndex vector search operations.
# This adapter wraps LlamaIndex functionality to implement the VectorSearchRepository port.

import logging
from typing import List, Optional
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from core.ports.repositories import VectorSearchRepository
from core.domain.models import DocumentChunk, SearchQuery, SearchResult, DocumentMetadata, SearchMode
from infrastructure.store.vector.faiss_store import get_faiss_vector_store
from infrastructure.store.vector.chroma_store import get_chroma_vector_store
from .mappers.llamaindex_mapper import LlamaIndexMapper

logger = logging.getLogger(__name__)


class LlamaIndexVectorAdapter(VectorSearchRepository):
    """
    Adapter that implements vector search using LlamaIndex.
    
    This adapter encapsulates all LlamaIndex-specific logic and provides
    a clean interface for the core domain to use.
    """
    
    def __init__(
        self, 
        storage_dir: str, 
        embedding_model: str, 
        vector_backend: str = "faiss"
    ):
        self.storage_dir = Path(storage_dir)
        self.embedding_model = embedding_model
        self.vector_backend = vector_backend.lower()
        self._index = None
        self._ensure_settings()
    
    def _ensure_settings(self):
        """Configure LlamaIndex global settings."""
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model)
    
    def _get_vector_store(self):
        """Factory method to get the appropriate vector store."""
        if self.vector_backend == "faiss":
            return get_faiss_vector_store()
        else:
            return get_chroma_vector_store()
    
    def _get_index(self) -> VectorStoreIndex:
        """Get or create the vector index."""
        if self._index is not None:
            return self._index
        
        persist_dir = self.storage_dir / "li_storage"
        
        if persist_dir.exists():
            # Load existing index
            vector_store = self._get_vector_store()
            storage_context = StorageContext.from_defaults(
                persist_dir=str(persist_dir),
                vector_store=vector_store
            )
            self._index = load_index_from_storage(storage_context=storage_context)
            logger.info(f"Loaded existing vector index from {persist_dir}")
        else:
            # Create new index
            vector_store = self._get_vector_store()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_documents([], storage_context=storage_context)
            storage_context.persist(persist_dir=str(persist_dir))
            logger.info(f"Created new vector index at {persist_dir}")
        
        return self._index
    
    async def index_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Index document chunks for vector search."""
        try:
            index = self._get_index()
            
            # Convert domain chunks to LlamaIndex documents using mapper
            documents = LlamaIndexMapper.domain_chunks_to_llama_documents(chunks)
            
            # Add documents to index
            for doc in documents:
                index.insert(doc)
            
            # Persist the updated index
            persist_dir = self.storage_dir / "li_storage"
            index.storage_context.persist(persist_dir=str(persist_dir))
            
            logger.info(f"Successfully indexed {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return False
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform vector similarity search."""
        try:
            index = self._get_index()
            
            # Create retriever
            retriever = index.as_retriever(similarity_top_k=query.top_k)
            
            # Execute search
            nodes = retriever.retrieve(query.text)
            
            # Convert results to domain format
            results = []
            for i, node in enumerate(nodes):
                metadata_dict = node.metadata or {}
                
                # Convert back to domain metadata
                metadata = DocumentMetadata(
                    doc_id=str(metadata_dict.get("doc_id", "unknown")),
                    chunk_id=str(metadata_dict.get("chunk_id", "")),
                    title=metadata_dict.get("title"),
                    page=metadata_dict.get("page"),
                    doc_type=metadata_dict.get("doc_type"),
                    faculty=metadata_dict.get("faculty"),
                    year=metadata_dict.get("year"),
                    subject=metadata_dict.get("subject"),
                    section=metadata_dict.get("section"),
                    subsection=metadata_dict.get("subsection"),
                    extra={k: v for k, v in metadata_dict.items() 
                          if k not in ["doc_id", "chunk_id", "title", "page", "doc_type", 
                                       "faculty", "year", "subject", "section", "subsection"]}
                )
                
                result = SearchResult(
                    text=node.get_text(),
                    metadata=metadata,
                    score=float(node.score or 0.0),
                    source_type="vector",
                    rank=i + 1
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    async def delete_document_vectors(self, doc_id: str) -> bool:
        """Remove vectors for a specific document."""
        try:
            index = self._get_index()
            
            # Note: LlamaIndex doesn't have built-in document deletion by doc_id
            # This is a limitation we need to work around
            # For now, we'll return True but this needs proper implementation
            logger.warning(f"Document deletion not fully implemented for LlamaIndex - doc_id: {doc_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document vectors: {e}")
            return False
