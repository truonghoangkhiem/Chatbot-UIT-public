# core/domain/search_service.py
#
# Description:
# Core domain service implementing the business logic for search operations.
# This service is completely independent of external frameworks and technologies.

import logging
from typing import List, Optional, Dict, Any
from .models import SearchQuery, SearchResponse, SearchResult, SearchMode, DocumentChunk
from ..ports.repositories import VectorSearchRepository, KeywordSearchRepository
from ..ports.services import RerankingService, FusionService, HighlightingService

logger = logging.getLogger(__name__)


class SearchService:
    """
    Core domain service for search operations.
    
    This service orchestrates search operations using various repositories and services
    through well-defined ports, ensuring the business logic remains independent of
    infrastructure concerns.
    """
    
    def __init__(
        self,
        vector_repository: VectorSearchRepository,
        keyword_repository: Optional[KeywordSearchRepository] = None,
        reranking_service: Optional[RerankingService] = None,
        fusion_service: Optional[FusionService] = None,
        highlighting_service: Optional[HighlightingService] = None
    ):
        self.vector_repository = vector_repository
        self.keyword_repository = keyword_repository
        self.reranking_service = reranking_service
        self.fusion_service = fusion_service
        self.highlighting_service = highlighting_service
    
    async def search(self, query: SearchQuery) -> SearchResponse:
        """
        Execute a search query using the appropriate strategy.
        
        Args:
            query: The search query containing all search parameters
            
        Returns:
            SearchResponse: Complete search results with metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Route to appropriate search strategy
            if query.search_mode == SearchMode.VECTOR:
                results = await self._vector_search(query)
            elif query.search_mode == SearchMode.BM25:
                results = await self._keyword_search(query)
            elif query.search_mode == SearchMode.HYBRID:
                results = await self._hybrid_search(query)
            else:
                raise ValueError(f"Unsupported search mode: {query.search_mode}")
            
            # Apply reranking if requested and available
            if query.use_rerank and self.reranking_service and self.reranking_service.is_available():
                # Use more candidates for reranking to improve accuracy
                rerank_top_k = min(len(results), query.top_k * 2)  # Get 2x more candidates for reranking
                results = await self.reranking_service.rerank(query.text, results, top_k=rerank_top_k)
                logger.info(f"Applied reranking to {len(results)} results (top_k={rerank_top_k})")
            
            # Apply highlighting if requested and available
            if query.highlight_matches and self.highlighting_service:
                results = await self.highlighting_service.highlight_matches(query.text, results)
            
            # Limit to requested number of results
            results = results[:query.top_k]
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return SearchResponse(
                results=results,
                total_hits=len(results),
                latency_ms=latency_ms,
                search_metadata={
                    "search_mode": query.search_mode.value,
                    "use_rerank": query.use_rerank,
                    "filters_applied": query.filters is not None
                }
            )
            
        except Exception as e:
            logger.error(f"Error in search execution: {e}")
            raise SearchError(f"Search failed: {str(e)}") from e
    
    async def _vector_search(self, query: SearchQuery) -> List[SearchResult]:
        """Execute vector-only search."""
        logger.info(f"Executing vector search for query: '{query.text[:50]}...'")
        
        # Increase candidate count for better reranking
        candidate_query = SearchQuery(
            text=query.text,
            top_k=max(query.top_k * 4, 16),
            search_mode=query.search_mode,
            filters=query.filters,
            include_char_spans=query.include_char_spans
        )
        
        results = await self.vector_repository.search(candidate_query)
        
        # Mark source type
        for result in results:
            result.source_type = "vector"
        
        return results
    
    async def _keyword_search(self, query: SearchQuery) -> List[SearchResult]:
        """Execute keyword/BM25-only search."""
        if not self.keyword_repository:
            raise SearchError("Keyword search not available - no keyword repository configured")
        
        logger.info(f"Executing keyword search for query: '{query.text[:50]}...'")
        
        # Increase candidate count for better reranking
        candidate_query = SearchQuery(
            text=query.text,
            top_k=max(query.top_k * 2, 12),
            search_mode=query.search_mode,
            filters=query.filters,
            include_char_spans=query.include_char_spans
        )
        
        results = await self.keyword_repository.search(candidate_query)
        
        # Mark source type
        for result in results:
            result.source_type = "bm25"
        
        return results
    
    async def _hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """Execute hybrid search combining vector and keyword results."""
        if not self.keyword_repository:
            logger.warning("Hybrid search requested but keyword repository not available, falling back to vector search")
            return await self._vector_search(query)
        
        if not self.fusion_service:
            raise SearchError("Hybrid search not available - no fusion service configured")
        
        logger.info(f"Executing hybrid search for query: '{query.text[:50]}...'")
        
        # Get more candidates for better fusion results
        candidate_count = max(query.top_k * 4, 20)
        
        # Execute both searches concurrently
        vector_query = SearchQuery(
            text=query.text,
            top_k=candidate_count,
            search_mode=SearchMode.VECTOR,
            filters=query.filters,
            include_char_spans=query.include_char_spans
        )
        
        keyword_query = SearchQuery(
            text=query.text,
            top_k=candidate_count,
            search_mode=SearchMode.BM25,
            filters=query.filters,
            include_char_spans=query.include_char_spans
        )
        
        # Execute searches
        vector_results = await self.vector_repository.search(vector_query)
        keyword_results = await self.keyword_repository.search(keyword_query)
        
        # Mark source types
        for result in vector_results:
            result.source_type = "vector"
        for result in keyword_results:
            result.source_type = "bm25"
        
        # Fuse results
        vector_weight = query.vector_weight or 0.5
        bm25_weight = query.bm25_weight or 0.5
        
        fused_results = await self.fusion_service.fuse_results(
            vector_results=vector_results,
            keyword_results=keyword_results,
            vector_weight=vector_weight,
            keyword_weight=bm25_weight
        )
        
        # Mark as fused
        for result in fused_results:
            result.source_type = "fused"
        
        logger.info(f"Fused {len(vector_results)} vector + {len(keyword_results)} keyword results into {len(fused_results)} final results")
        
        return fused_results
    
    async def index_documents(self, chunks: List[DocumentChunk]) -> bool:
        """
        Index document chunks in all available repositories.
        
        Args:
            chunks: List of document chunks to index
            
        Returns:
            bool: True if indexing succeeded in at least one repository
        """
        success_count = 0
        
        # Index in vector repository
        try:
            if await self.vector_repository.index_documents(chunks):
                success_count += 1
                logger.info(f"Successfully indexed {len(chunks)} chunks in vector repository")
        except Exception as e:
            logger.error(f"Failed to index chunks in vector repository: {e}")
        
        # Index in keyword repository if available
        if self.keyword_repository:
            try:
                if await self.keyword_repository.index_documents(chunks):
                    success_count += 1
                    logger.info(f"Successfully indexed {len(chunks)} chunks in keyword repository")
            except Exception as e:
                logger.error(f"Failed to index chunks in keyword repository: {e}")
        
        return success_count > 0
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from all repositories.
        
        Args:
            doc_id: ID of the document to delete
            
        Returns:
            bool: True if deletion succeeded in at least one repository
        """
        success_count = 0
        
        # Delete from vector repository
        try:
            if await self.vector_repository.delete_document_vectors(doc_id):
                success_count += 1
                logger.info(f"Successfully deleted document {doc_id} from vector repository")
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id} from vector repository: {e}")
        
        # Delete from keyword repository if available
        if self.keyword_repository:
            try:
                if await self.keyword_repository.delete_document_index(doc_id):
                    success_count += 1
                    logger.info(f"Successfully deleted document {doc_id} from keyword repository")
            except Exception as e:
                logger.error(f"Failed to delete document {doc_id} from keyword repository: {e}")
        
        return success_count > 0


class SearchError(Exception):
    """Domain exception for search-related errors."""
    pass
