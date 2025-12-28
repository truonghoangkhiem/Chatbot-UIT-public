# app/api/schemas/search.py
#
# Description:
# This file defines the Pydantic models for the core search functionality,
# including the search request, a single search result (hit), and the final search response.

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from .common import SourceMeta, CitationSpan, CharacterSpan

class SearchRequest(BaseModel):
    """
    Enhanced request body for the /search endpoint with field filters.
    """
    query: str
    top_k: int = 8
    filters: Optional[Dict[str, Any]] = None  # Legacy metadata filters
    need_citation: bool = True
    use_rerank: bool = True
    
    # Hybrid search options
    use_hybrid: Optional[bool] = None  # Override global setting
    search_mode: Optional[str] = "hybrid"  # "vector", "bm25", "hybrid"
    bm25_weight: Optional[float] = None  # Override default BM25 weight
    vector_weight: Optional[float] = None  # Override default vector weight
    
    # Field-specific filters
    doc_types: Optional[List[str]] = None  # e.g., ["syllabus", "regulation"]
    faculties: Optional[List[str]] = None  # e.g., ["CNTT", "KHTN", "CTDA"]
    years: Optional[List[int]] = None  # e.g., [2023, 2024]
    subjects: Optional[List[str]] = None  # e.g., ["SE101", "CS201"]
    language: Optional[str] = "vi"  # Document language
    
    # Citation options
    include_char_spans: bool = True
    highlight_matches: bool = True

class SearchHit(BaseModel):
    """
    Enhanced structure of a single search result item.
    """
    text: str
    title: Optional[str] = None
    score: float
    meta: SourceMeta
    citation: Optional[CitationSpan] = None
    rerank_score: Optional[float] = None
    
    # Document classification fields
    doc_type: Optional[str] = None  # e.g., "syllabus", "regulation"
    faculty: Optional[str] = None   # e.g., "CNTT", "KHTN"
    year: Optional[int] = None      # Academic year
    subject: Optional[str] = None   # Subject/course code
    language: Optional[str] = None  # Document language
    
    # Search metadata
    source_type: Optional[str] = None  # "vector", "bm25", "fused"
    bm25_score: Optional[float] = None  # Original BM25 score
    vector_score: Optional[float] = None  # Original vector score
    fusion_rank: Optional[int] = None  # Rank after fusion
    
    # Enhanced citation data
    char_spans: Optional[List[CharacterSpan]] = None
    highlighted_text: Optional[List[str]] = None
    highlighted_title: Optional[List[str]] = None

class SearchFacet(BaseModel):
    """Structure for search facets/aggregations."""
    key: str
    doc_count: int

class SearchResponse(BaseModel):
    """
    Enhanced response structure for the /search endpoint.
    """
    hits: List[SearchHit]
    total_hits: int
    latency_ms: int
    
    # Faceted search results
    facets: Optional[Dict[str, List[SearchFacet]]] = None
    
    # Search metadata
    search_metadata: Optional[Dict[str, Any]] = None