"""
RAG service adapter for integrating with the RAG services.

This adapter provides integration with the RAG system following the
Ports & Adapters architecture pattern.

Enhanced with filter support for:
- doc_types: Document type filters (e.g., ["syllabus", "regulation"])
- faculties: Faculty filters (e.g., ["CNTT", "KHTN"])
- years: Academic year filters (e.g., [2023, 2024])
- subjects: Subject/course code filters (e.g., ["SE101", "CS201"])
"""

import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
from ..ports.agent_ports import RAGServicePort


class RAGFilters:
    """Filters for RAG search requests."""
    
    def __init__(
        self,
        doc_types: Optional[List[str]] = None,
        faculties: Optional[List[str]] = None,
        years: Optional[List[int]] = None,
        subjects: Optional[List[str]] = None,
        language: str = "vi"
    ):
        self.doc_types = doc_types
        self.faculties = faculties
        self.years = years
        self.subjects = subjects
        self.language = language
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filters to dictionary for API payload."""
        result = {}
        if self.doc_types:
            result["doc_types"] = self.doc_types
        if self.faculties:
            result["faculties"] = self.faculties
        if self.years:
            result["years"] = self.years
        if self.subjects:
            result["subjects"] = self.subjects
        if self.language:
            result["language"] = self.language
        return result
    
    def is_empty(self) -> bool:
        """Check if no filters are set."""
        return not any([self.doc_types, self.faculties, self.years, self.subjects])


class RAGServiceAdapter(RAGServicePort):
    """
    Adapter for RAG service integration.
    
    This adapter implements the RAGServicePort interface to provide
    communication with the RAG services.
    
    Features:
    - Field-specific filters (doc_types, faculties, years, subjects)
    - Hybrid search with configurable modes
    - Citation support with character spans
    - Retry mechanism for resilience
    """
    
    def __init__(
        self,
        rag_service_url: str = "http://localhost:8001",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the RAG service adapter.
        
        Args:
            rag_service_url: Base URL for the RAG service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.rag_service_url = rag_service_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={"Content-Type": "application/json"}
            )
        return self._session
    
    async def retrieve_context(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[RAGFilters] = None,
        search_mode: str = "hybrid",
        use_rerank: bool = True,
        need_citation: bool = True,
        include_char_spans: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context for a query using the RAG system.
        
        Args:
            query: The user query to search for relevant documents
            top_k: Number of top relevant documents to retrieve
            filters: Optional RAGFilters object with doc_types, faculties, years, subjects
            search_mode: Search mode - "vector", "bm25", or "hybrid" (default)
            use_rerank: Whether to use reranking (default True)
            need_citation: Whether to include citation info (default True)
            include_char_spans: Whether to include character spans for precise citation
            
        Returns:
            Dictionary containing retrieved documents with metadata and citations
        """
        session = await self._get_session()
        
        # OPTIMIZATION: Limit reranking to top 5 documents for performance
        # Instead of reranking all retrieved docs, only rerank the most promising ones
        rerank_top_n = min(5, top_k) if use_rerank else 0
        
        # Build payload with all supported parameters
        payload = {
            "query": query,
            "top_k": top_k,
            "search_mode": search_mode,
            "use_rerank": use_rerank,
            "rerank_top_n": rerank_top_n,  # Limit reranking scope
            "need_citation": need_citation,
            "include_char_spans": include_char_spans,
            "highlight_matches": True
        }
        
        # Add filters if provided
        if filters and not filters.is_empty():
            filter_dict = filters.to_dict()
            if filter_dict.get("doc_types"):
                payload["doc_types"] = filter_dict["doc_types"]
            if filter_dict.get("faculties"):
                payload["faculties"] = filter_dict["faculties"]
            if filter_dict.get("years"):
                payload["years"] = filter_dict["years"]
            if filter_dict.get("subjects"):
                payload["subjects"] = filter_dict["subjects"]
            if filter_dict.get("language"):
                payload["language"] = filter_dict["language"]
        
        for attempt in range(self.max_retries):
            try:
                import logging
                import json as json_module
                logger = logging.getLogger(__name__)
                logger.info(f"🔍 RAG Adapter: Sending request to {self.rag_service_url}/v1/search")
                logger.info(f"   Full payload: {json_module.dumps(payload, ensure_ascii=False)}")
                
                async with session.post(
                    f"{self.rag_service_url}/v1/search",
                    json=payload
                ) as response:
                    
                    logger.info(f"   Response status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"   Full response keys: {list(data.keys())}")
                        
                        # Transform response to standard format with enhanced citation data
                        hits = data.get("hits", [])
                        logger.info(f"   Got {len(hits)} hits from RAG service")
                        
                        # Process hits to include citation information
                        processed_hits = []
                        for hit in hits:
                            processed_hit = {
                                "text": hit.get("text", ""),
                                "title": hit.get("title", ""),
                                "score": hit.get("score", 0.0),
                                "meta": hit.get("meta", {}),
                                "rerank_score": hit.get("rerank_score"),
                                # Document classification
                                "doc_type": hit.get("doc_type"),
                                "faculty": hit.get("faculty"),
                                "year": hit.get("year"),
                                "subject": hit.get("subject"),
                                # Search metadata
                                "source_type": hit.get("source_type"),
                                "bm25_score": hit.get("bm25_score"),
                                "vector_score": hit.get("vector_score"),
                                # Enhanced citation data
                                "citation": hit.get("citation"),
                                "char_spans": hit.get("char_spans"),
                                "highlighted_text": hit.get("highlighted_text"),
                                "highlighted_title": hit.get("highlighted_title")
                            }
                            processed_hits.append(processed_hit)
                        
                        return {
                            "query": query,
                            "retrieved_documents": processed_hits,
                            "search_metadata": {
                                "total_results": data.get("total_hits", len(hits)),
                                "search_mode": search_mode,
                                "processing_time": data.get("latency_ms"),
                                "top_k": top_k,
                                "filters_applied": filters.to_dict() if filters else None,
                                "facets": data.get("facets"),
                                "search_metadata": data.get("search_metadata")
                            },
                            "relevance_scores": [
                                doc.get("score", 0.0) 
                                for doc in processed_hits
                            ]
                        }
                    
                    elif response.status == 503:  # Service unavailable
                        if attempt < self.max_retries - 1:
                            wait_time = 2 ** attempt
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise Exception(f"RAG service unavailable after {self.max_retries} attempts")
                    
                    else:
                        error_text = await response.text()
                        raise Exception(f"RAG service error {response.status}: {error_text}")
            
            except aiohttp.ClientError as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Network error communicating with RAG service: {str(e)}")
        
        raise Exception("Max retries exceeded for RAG service")
    
    async def health_check(self) -> bool:
        """
        Check if the RAG service is healthy and available.
        
        Returns:
            True if the service is healthy, False otherwise
        """
        try:
            session = await self._get_session()
            
            async with session.get(
                f"{self.rag_service_url}/v1/health"
            ) as response:
                return response.status == 200
        
        except Exception:
            return False
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()