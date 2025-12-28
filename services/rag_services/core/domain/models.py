# core/domain/models.py
#
# Description:
# Core domain models representing the business entities in the RAG system.
# These models are framework-agnostic and contain pure business logic.

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class SearchMode(Enum):
    """Enumeration of supported search modes."""
    VECTOR = "vector"
    BM25 = "bm25" 
    HYBRID = "hybrid"


class DocumentLanguage(Enum):
    """Supported document languages."""
    VIETNAMESE = "vi"
    ENGLISH = "en"


@dataclass
class CharacterSpan:
    """Represents a character span within a document for precise citation."""
    start: int
    end: int
    text: str
    type: str = "content"  # "content", "sentence", "paragraph"


@dataclass
class DocumentMetadata:
    """Metadata associated with a document."""
    doc_id: str
    chunk_id: Optional[str] = None
    title: Optional[str] = None
    page: Optional[int] = None
    doc_type: Optional[str] = None  # "syllabus", "regulation", etc.
    faculty: Optional[str] = None   # "CNTT", "KHTN", etc.
    year: Optional[int] = None      # Academic year
    subject: Optional[str] = None   # Subject/course code
    language: DocumentLanguage = DocumentLanguage.VIETNAMESE
    section: Optional[str] = None
    subsection: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchFilters:
    """Filters to apply during search."""
    doc_types: Optional[List[str]] = None
    faculties: Optional[List[str]] = None
    years: Optional[List[int]] = None
    subjects: Optional[List[str]] = None
    language: Optional[DocumentLanguage] = None
    metadata_filters: Optional[Dict[str, Any]] = None


@dataclass
class SearchQuery:
    """Represents a search query in the domain."""
    text: str
    top_k: int = 8
    search_mode: SearchMode = SearchMode.HYBRID
    use_rerank: bool = True
    bm25_weight: Optional[float] = None
    vector_weight: Optional[float] = None
    filters: Optional[SearchFilters] = None
    include_char_spans: bool = True
    highlight_matches: bool = True


@dataclass
class RerankingMetadata:
    """Metadata for reranking operations."""
    original_rank: int
    original_score: float
    rerank_score: float
    confidence: Optional[float] = None
    model_name: Optional[str] = None
    processing_time_ms: Optional[int] = None


@dataclass
class SearchResult:
    """Represents a single search result from the domain perspective."""
    text: str
    metadata: DocumentMetadata
    score: float
    source_type: str  # "vector", "bm25", "fused", etc.
    rank: Optional[int] = None
    char_spans: Optional[List[CharacterSpan]] = None
    highlighted_text: Optional[List[str]] = None
    highlighted_title: Optional[List[str]] = None
    # Detailed scores for transparency
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    rerank_score: Optional[float] = None
    reranking_metadata: Optional[RerankingMetadata] = None


@dataclass
class SearchResponse:
    """Complete search response from the domain."""
    results: List[SearchResult]
    total_hits: int
    latency_ms: int
    facets: Optional[Dict[str, List[Dict[str, Any]]]] = None
    search_metadata: Optional[Dict[str, Any]] = None


@dataclass
class Document:
    """Represents a document in the system."""
    text: str
    metadata: DocumentMetadata
    
    
@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    text: str
    metadata: DocumentMetadata
    chunk_index: int
    char_spans: Optional[List[CharacterSpan]] = None
