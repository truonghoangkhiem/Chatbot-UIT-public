# adapters/mappers/search_mappers.py
#
# Description:
# Mappers to convert between API schemas (FastAPI/Pydantic models) and domain models.
# This ensures the core domain remains independent of web framework specifics.

from typing import List, Optional

from app.api.schemas.search import SearchRequest as ApiSearchRequest, SearchHit as ApiSearchHit, SearchResponse as ApiSearchResponse
from app.api.schemas.common import SourceMeta, CitationSpan, CharacterSpan as ApiCharacterSpan

from core.domain.models import (
    SearchQuery, SearchResult, SearchResponse, SearchMode, SearchFilters,
    DocumentLanguage, CharacterSpan as DomainCharacterSpan
)


class SearchMapper:
    """
    Mapper class for converting between API schemas and domain models.
    
    This mapper acts as an anti-corruption layer, ensuring that changes
    in the API layer don't affect the core domain logic.
    """
    
    @staticmethod
    def api_request_to_domain_query(api_request: ApiSearchRequest) -> SearchQuery:
        """Convert API SearchRequest to domain SearchQuery."""
        
        # Map search mode
        search_mode = SearchMode.HYBRID
        if api_request.search_mode:
            if api_request.search_mode == "vector":
                search_mode = SearchMode.VECTOR
            elif api_request.search_mode == "bm25":
                search_mode = SearchMode.BM25
            elif api_request.search_mode == "hybrid":
                search_mode = SearchMode.HYBRID
        
        # Map language
        language = None
        if api_request.language:
            language = DocumentLanguage.VIETNAMESE if api_request.language == "vi" else DocumentLanguage.ENGLISH
        
        # Map filters
        filters = None
        if (api_request.doc_types or api_request.faculties or api_request.years or 
            api_request.subjects or api_request.language or api_request.filters):
            filters = SearchFilters(
                doc_types=api_request.doc_types,
                faculties=api_request.faculties,
                years=api_request.years,
                subjects=api_request.subjects,
                language=language,
                metadata_filters=api_request.filters
            )
        
        return SearchQuery(
            text=api_request.query,
            top_k=api_request.top_k,
            search_mode=search_mode,
            use_rerank=api_request.use_rerank,
            bm25_weight=api_request.bm25_weight,
            vector_weight=api_request.vector_weight,
            filters=filters,
            include_char_spans=api_request.include_char_spans,
            highlight_matches=api_request.highlight_matches
        )
    
    @staticmethod
    def domain_response_to_api_response(domain_response: SearchResponse) -> ApiSearchResponse:
        """Convert domain SearchResponse to API SearchResponse."""
        
        api_hits = []
        for result in domain_response.results:
            api_hit = SearchMapper._domain_result_to_api_hit(result)
            api_hits.append(api_hit)
        
        return ApiSearchResponse(
            hits=api_hits,
            total_hits=domain_response.total_hits,
            latency_ms=domain_response.latency_ms,
            facets=domain_response.facets,
            search_metadata=domain_response.search_metadata
        )
    
    @staticmethod
    def _domain_result_to_api_hit(domain_result: SearchResult) -> ApiSearchHit:
        """Convert domain SearchResult to API SearchHit."""
        
        # Convert character spans
        char_spans = None
        if domain_result.char_spans:
            char_spans = [
                ApiCharacterSpan(
                    start=span.start,
                    end=span.end,
                    text=span.text,
                    type=span.type
                )
                for span in domain_result.char_spans
            ]
        
        # Create source metadata
        source_meta = SourceMeta(
            doc_id=domain_result.metadata.doc_id,
            chunk_id=domain_result.metadata.chunk_id,
            page=domain_result.metadata.page,
            extra=domain_result.metadata.extra
        )
        
        # Create citation
        citation = CitationSpan(
            doc_id=domain_result.metadata.doc_id,
            chunk_id=domain_result.metadata.chunk_id,
            page=domain_result.metadata.page,
            char_spans=char_spans,
            section=domain_result.metadata.section,
            subsection=domain_result.metadata.subsection,
            highlighted_text=domain_result.highlighted_text
        )
        
        return ApiSearchHit(
            text=domain_result.text,
            title=domain_result.metadata.title,
            score=domain_result.score,
            meta=source_meta,
            citation=citation,
            rerank_score=domain_result.rerank_score,
            
            # Document classification
            doc_type=domain_result.metadata.doc_type,
            faculty=domain_result.metadata.faculty,
            year=domain_result.metadata.year,
            subject=domain_result.metadata.subject,
            language=domain_result.metadata.language.value if domain_result.metadata.language else "vi",
            
            # Search metadata
            source_type=domain_result.source_type,
            bm25_score=domain_result.bm25_score,
            vector_score=domain_result.vector_score,
            fusion_rank=domain_result.rank,
            
            # Enhanced citation data
            char_spans=char_spans,
            highlighted_text=domain_result.highlighted_text,
            highlighted_title=domain_result.highlighted_title
        )
