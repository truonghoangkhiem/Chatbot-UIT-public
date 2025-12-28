"""
Mapper for converting between domain models and LlamaIndex format.

This mapper isolates LlamaIndex-specific formatting from the adapter logic.
"""

from typing import List, Dict, Any
from llama_index.core import Document as LlamaDocument

from core.domain.models import DocumentChunk, SearchResult, DocumentMetadata


class LlamaIndexMapper:
    """Handles conversion between domain models and LlamaIndex format."""
    
    @staticmethod
    def domain_chunk_to_llama_document(chunk: DocumentChunk) -> LlamaDocument:
        """
        Convert domain DocumentChunk to LlamaIndex Document.
        
        Args:
            chunk: Domain document chunk
            
        Returns:
            LlamaIndex Document
        """
        # Convert only essential metadata, avoiding LlamaIndex-specific fields
        metadata = {
            "doc_id": chunk.metadata.doc_id,
            "title": chunk.metadata.title or "",
            "page": chunk.metadata.page or 0,
            "chunk_index": chunk.chunk_index,
        }
        
        # Add optional fields only if they exist
        if chunk.metadata.chunk_id:
            metadata["chunk_id"] = chunk.metadata.chunk_id
        if chunk.metadata.doc_type:
            metadata["doc_type"] = chunk.metadata.doc_type
        if chunk.metadata.faculty:
            metadata["faculty"] = chunk.metadata.faculty
        if chunk.metadata.year:
            metadata["year"] = chunk.metadata.year
        if chunk.metadata.subject:
            metadata["subject"] = chunk.metadata.subject
        if chunk.metadata.language:
            metadata["language"] = chunk.metadata.language.value
        if chunk.metadata.section:
            metadata["section"] = chunk.metadata.section
        if chunk.metadata.subsection:
            metadata["subsection"] = chunk.metadata.subsection
            
        # Add extra metadata conservatively
        for key, value in chunk.metadata.extra.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
        
        return LlamaDocument(
            text=chunk.text,
            metadata=metadata
        )
    
    @staticmethod
    def domain_chunks_to_llama_documents(chunks: List[DocumentChunk]) -> List[LlamaDocument]:
        """Convert multiple domain chunks to LlamaIndex documents."""
        return [
            LlamaIndexMapper.domain_chunk_to_llama_document(chunk) 
            for chunk in chunks
        ]
    
    @staticmethod
    def llama_node_to_domain_result(node, score: float = 0.0) -> Dict[str, Any]:
        """
        Convert LlamaIndex node to domain-compatible result.
        
        Args:
            node: LlamaIndex node
            score: Relevance score
            
        Returns:
            Domain-compatible document dict
        """
        # Extract basic information
        content = getattr(node, 'text', '') or str(node)
        metadata = getattr(node, 'metadata', {}) or {}
        
        # Build domain-compatible result
        return {
            "content": content,
            "title": metadata.get("title", ""),
            "score": score,
            "metadata": {
                "doc_id": metadata.get("doc_id", ""),
                "chunk_id": metadata.get("chunk_id", ""),
                "page": metadata.get("page", 0),
                "doc_type": metadata.get("doc_type", ""),
                "faculty": metadata.get("faculty", ""),
                "year": metadata.get("year"),
                "subject": metadata.get("subject", ""),
                "language": metadata.get("language", "vi"),
                "section": metadata.get("section", ""),
                "subsection": metadata.get("subsection", ""),
                "chunk_index": metadata.get("chunk_index", 0)
            }
        }