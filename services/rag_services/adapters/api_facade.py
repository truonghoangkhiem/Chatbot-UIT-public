# adapters/api_facade.py
#
# Description:
# API Facade that provides a clean interface for the web layer to interact with the domain.
# This facade handles all the mapping between API schemas and domain models.

import logging
from typing import List

from app.api.schemas.search import SearchRequest, SearchResponse
from core.container import get_search_service
from core.domain.models import DocumentChunk
from adapters.mappers.search_mappers import SearchMapper

logger = logging.getLogger(__name__)


class SearchApiFacade:
    """
    Facade for search operations that bridges the API layer and domain layer.
    
    This facade:
    1. Converts API models to domain models
    2. Delegates to core domain services
    3. Converts domain responses back to API models
    4. Handles any API-specific concerns (logging, error handling, etc.)
    """
    
    def __init__(self):
        self.search_service = get_search_service()
    
    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Execute a search request.
        
        Args:
            request: API search request
            
        Returns:
            SearchResponse: API search response
        """
        try:
            # Convert API request to domain query
            domain_query = SearchMapper.api_request_to_domain_query(request)
            
            logger.info(f"Executing search: mode={domain_query.search_mode.value}, "
                       f"query='{domain_query.text[:50]}...', top_k={domain_query.top_k}")
            
            # Execute search in domain layer
            domain_response = await self.search_service.search(domain_query)
            
            # Convert domain response to API response
            api_response = SearchMapper.domain_response_to_api_response(domain_response)
            
            logger.info(f"Search completed: {domain_response.total_hits} results in {domain_response.latency_ms}ms")
            
            return api_response
            
        except Exception as e:
            logger.error(f"Error in search facade: {e}")
            # Return empty response on error to maintain API contract
            from app.api.schemas.search import SearchResponse
            return SearchResponse(
                hits=[],
                total_hits=0,
                latency_ms=0,
                search_metadata={"error": str(e)}
            )
    
    async def index_documents(self, chunks: List[DocumentChunk]) -> bool:
        """
        Index document chunks.
        
        Args:
            chunks: List of document chunks to index
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Indexing {len(chunks)} document chunks")
            
            success = await self.search_service.index_documents(chunks)
            
            if success:
                logger.info(f"Successfully indexed {len(chunks)} chunks")
            else:
                logger.error(f"Failed to index {len(chunks)} chunks")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in indexing facade: {e}")
            return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from all indexes.
        
        Args:
            doc_id: ID of the document to delete
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Deleting document: {doc_id}")
            
            success = await self.search_service.delete_document(doc_id)
            
            if success:
                logger.info(f"Successfully deleted document: {doc_id}")
            else:
                logger.error(f"Failed to delete document: {doc_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in document deletion facade: {e}")
            return False


# Global facade instance for easy access
_search_facade = None


def get_search_facade() -> SearchApiFacade:
    """Get the global search facade instance."""
    global _search_facade
    if _search_facade is None:
        _search_facade = SearchApiFacade()
    return _search_facade
