# app/api/v1/routes/search.py
#
# Description:
# This module implements the core search endpoint for the RAG service.
# Uses the clean Ports & Adapters architecture exclusively.

from fastapi import APIRouter, HTTPException
from app.api.schemas.search import SearchRequest, SearchResponse
from adapters.api_facade import get_search_facade
import logging

# Create an API router for the search functionality
router = APIRouter(tags=["search"])
logger = logging.getLogger(__name__)

@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    Endpoint to perform a search query against the indexed documents.
    Supports vector, BM25, and hybrid search modes using clean architecture.

    Args:
        req (SearchRequest): The search request containing the query and other parameters.

    Returns:
        SearchResponse: Complete search response with hits, latency, and metadata.
    """
    try:
        # Use clean Ports & Adapters architecture exclusively
        search_facade = get_search_facade()
        return await search_facade.search(req)
        
    except Exception as e:
        logger.error(f"Search request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search service temporarily unavailable: {str(e)}"
        )