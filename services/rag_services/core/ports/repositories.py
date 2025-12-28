# core/ports/repositories.py
#
# Description:
# Port interfaces for data access and storage operations.
# These abstractions allow the core domain to remain independent of specific storage technologies.

from abc import ABC, abstractmethod
from typing import List, Optional
from ..domain.models import Document, DocumentChunk, SearchQuery, SearchResult, DocumentMetadata


class DocumentRepository(ABC):
    """Port for document storage and retrieval operations."""
    
    @abstractmethod
    async def store_document(self, document: Document) -> str:
        """Store a document and return its ID."""
        pass
    
    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        pass
    
    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        pass
    
    @abstractmethod
    async def list_documents(self, filters: Optional[dict] = None) -> List[DocumentMetadata]:
        """List document metadata with optional filters."""
        pass


class VectorSearchRepository(ABC):
    """Port for vector-based similarity search operations."""
    
    @abstractmethod
    async def index_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Index document chunks for vector search."""
        pass
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform vector similarity search."""
        pass
    
    @abstractmethod
    async def delete_document_vectors(self, doc_id: str) -> bool:
        """Remove vectors for a specific document."""
        pass


class KeywordSearchRepository(ABC):
    """Port for keyword/BM25-based search operations."""
    
    @abstractmethod
    async def index_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Index document chunks for keyword search."""
        pass
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform keyword-based search."""
        pass
    
    @abstractmethod
    async def delete_document_index(self, doc_id: str) -> bool:
        """Remove index entries for a specific document."""
        pass
    
    @abstractmethod
    async def get_facets(self, query: SearchQuery) -> dict:
        """Get search facets/aggregations."""
        pass
