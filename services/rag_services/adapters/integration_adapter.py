# adapters/integration_adapter.py
#
# Description:
# Integration adapter to gradually migrate from old architecture to new clean architecture.
# This provides a smooth transition path while maintaining backward compatibility.

import logging
import asyncio
from typing import List
from concurrent.futures import ThreadPoolExecutor

from app.api.schemas.search import SearchRequest, SearchHit
from core.container import get_search_service
from core.domain.models import SearchQuery, SearchMode, SearchFilters, DocumentLanguage
from adapters.mappers.search_mappers import SearchMapper

logger = logging.getLogger(__name__)


class IntegrationSearchAdapter:
    """
    Integration adapter that provides a synchronous interface
    while using the new clean architecture internally.
    
    This adapter handles the complexity of bridging sync/async boundaries
    and provides a gradual migration path.
    """
    
    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._search_service = None
    
    def _get_search_service(self):
        """Lazy initialization of search service."""
        if self._search_service is None:
            self._search_service = get_search_service()
        return self._search_service
    
    def search(self, req: SearchRequest) -> List[SearchHit]:
        """
        Synchronous search interface that delegates to async implementation.
        
        Args:
            req: API search request
            
        Returns:
            List of search hits
        """
        try:
            # Convert to domain query
            domain_query = SearchMapper.api_request_to_domain_query(req)
            
            logger.info(f"Integration adapter executing search: {domain_query.search_mode.value}")
            
            # Execute async search in thread pool
            future = self._executor.submit(self._run_async_search, domain_query)
            domain_response = future.result(timeout=30)  # 30 second timeout
            
            # Convert back to API format
            api_response = SearchMapper.domain_response_to_api_response(domain_response)
            
            logger.info(f"Integration search completed: {len(api_response.hits)} results")
            return api_response.hits
            
        except Exception as e:
            logger.error(f"Error in integration search adapter: {e}")
            return []
    
    def _run_async_search(self, domain_query):
        """Run async search in a new event loop."""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                search_service = self._get_search_service()
                return loop.run_until_complete(search_service.search(domain_query))
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error in async search execution: {e}")
            # Return empty response
            from core.domain.models import SearchResponse
            return SearchResponse(
                results=[],
                total_hits=0,
                latency_ms=0,
                search_metadata={"error": str(e)}
            )


# Global adapter instance
_integration_adapter = None

def get_integration_adapter() -> IntegrationSearchAdapter:
    """Get the global integration adapter instance."""
    global _integration_adapter
    if _integration_adapter is None:
        _integration_adapter = IntegrationSearchAdapter()
    return _integration_adapter
