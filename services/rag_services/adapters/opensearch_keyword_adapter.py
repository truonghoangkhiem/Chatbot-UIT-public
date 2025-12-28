# adapters/opensearch_keyword_adapter.py
#
# Description:
# Adapter implementation for OpenSearch keyword/BM25 search operations.
# This adapter wraps OpenSearch functionality to implement the KeywordSearchRepository port.

import logging
from typing import List, Optional, Dict, Any

from core.ports.repositories import KeywordSearchRepository
from core.domain.models import (
    DocumentChunk, SearchQuery, SearchResult, DocumentMetadata, 
    CharacterSpan, DocumentLanguage
)

logger = logging.getLogger(__name__)


class OpenSearchKeywordAdapter(KeywordSearchRepository):
    """
    Adapter that implements keyword search using OpenSearch.
    
    This adapter encapsulates all OpenSearch-specific logic and provides
    a clean interface for the core domain to use.
    """
    
    def __init__(self, opensearch_client):
        self.client = opensearch_client
    
    async def index_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Index document chunks for keyword search."""
        try:
            # Convert domain chunks to OpenSearch format
            documents = []
            for chunk in chunks:
                doc = {
                    "doc_id": chunk.metadata.doc_id,
                    "chunk_id": chunk.metadata.chunk_id or "",
                    "text": chunk.text,
                    "title": chunk.metadata.title or "",
                    "page": chunk.metadata.page,
                    "doc_type": chunk.metadata.doc_type,
                    "faculty": chunk.metadata.faculty,
                    "year": chunk.metadata.year,
                    "subject": chunk.metadata.subject,
                    "language": chunk.metadata.language.value if chunk.metadata.language else "vi",
                    "section": chunk.metadata.section,
                    "subsection": chunk.metadata.subsection,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata.extra
                }
                
                # Add character spans if available
                if chunk.char_spans:
                    doc["char_spans"] = [
                        {
                            "start": span.start,
                            "end": span.end,
                            "text": span.text,
                            "type": span.type
                        }
                        for span in chunk.char_spans
                    ]
                
                documents.append(doc)
            
            # Bulk index to OpenSearch
            success = self._bulk_index_documents(documents)
            
            if success:
                logger.info(f"Successfully indexed {len(chunks)} chunks in OpenSearch")
            
            return success
            
        except Exception as e:
            logger.error(f"Error indexing documents in OpenSearch: {e}")
            return False
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform keyword-based search."""
        try:
            # Build OpenSearch query
            search_params = self._build_search_params(query)
            
            # Execute search
            os_results = self._execute_search(search_params)
            
            # Convert results to domain format
            results = []
            for i, os_result in enumerate(os_results):
                metadata = self._convert_opensearch_metadata(os_result)
                
                # Extract character spans
                char_spans = None
                if query.include_char_spans and "char_spans" in os_result:
                    char_spans = [
                        CharacterSpan(
                            start=span.get("start", 0),
                            end=span.get("end", 0),
                            text=span.get("text", ""),
                            type=span.get("type", "content")
                        )
                        for span in os_result["char_spans"]
                    ]
                
                result = SearchResult(
                    text=os_result["text"],
                    metadata=metadata,
                    score=float(os_result.get("bm25_score", 0.0)),
                    source_type="bm25",
                    rank=i + 1,
                    char_spans=char_spans,
                    highlighted_text=os_result.get("highlighted_text"),
                    highlighted_title=os_result.get("highlighted_title"),
                    bm25_score=float(os_result.get("bm25_score", 0.0))
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in OpenSearch keyword search: {e}")
            return []
    
    async def delete_document_index(self, doc_id: str) -> bool:
        """Remove index entries for a specific document."""
        try:
            # Use OpenSearch delete by query
            delete_query = {
                "query": {
                    "term": {"doc_id": doc_id}
                }
            }
            
            success = self._delete_by_query(delete_query)
            
            if success:
                logger.info(f"Successfully deleted document {doc_id} from OpenSearch index")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting document from OpenSearch: {e}")
            return False
    
    async def get_facets(self, query: SearchQuery) -> dict:
        """Get search facets/aggregations."""
        try:
            # Build aggregation query
            agg_params = self._build_aggregation_params(query)
            
            # Execute aggregation
            facets = self._execute_aggregation(agg_params)
            
            return facets
            
        except Exception as e:
            logger.error(f"Error getting facets from OpenSearch: {e}")
            return {}
    
    def _build_search_params(self, query: SearchQuery) -> dict:
        """Build OpenSearch query parameters from domain query."""
        params = {
            "query": query.text,
            "size": query.top_k,
            "include_char_spans": query.include_char_spans,
            "language": query.filters.language.value if query.filters and query.filters.language else "vi"
        }
        
        # Add filters if present
        if query.filters:
            if query.filters.doc_types:
                params["doc_types"] = query.filters.doc_types
            if query.filters.faculties:
                params["faculties"] = query.filters.faculties
            if query.filters.years:
                params["years"] = query.filters.years
            if query.filters.subjects:
                params["subjects"] = query.filters.subjects
            if query.filters.metadata_filters:
                params["filters"] = query.filters.metadata_filters
        
        return params
    
    def _convert_opensearch_metadata(self, os_result: dict) -> DocumentMetadata:
        """Convert OpenSearch result to domain metadata."""
        language = DocumentLanguage.VIETNAMESE
        if os_result.get("language") == "en":
            language = DocumentLanguage.ENGLISH
        
        return DocumentMetadata(
            doc_id=os_result["doc_id"],
            chunk_id=os_result.get("chunk_id"),
            title=os_result.get("title"),
            page=os_result.get("page"),
            doc_type=os_result.get("doc_type"),
            faculty=os_result.get("faculty"),
            year=os_result.get("year"),
            subject=os_result.get("subject"),
            language=language,
            section=os_result.get("section"),
            subsection=os_result.get("subsection"),
            extra=os_result.get("metadata", {})
        )
    
    def _bulk_index_documents(self, documents: List[dict]) -> bool:
        """Bulk index documents to OpenSearch."""
        # Delegate to the actual OpenSearch client implementation
        # This method should be implemented based on your OpenSearch client
        try:
            if hasattr(self.client, 'bulk_index'):
                return self.client.bulk_index(documents)
            else:
                # Fallback implementation
                logger.warning("bulk_index method not available on OpenSearch client")
                return False
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            return False
    
    def _execute_search(self, params: dict) -> List[dict]:
        """Execute search query on OpenSearch."""
        # Delegate to the actual OpenSearch client implementation
        try:
            return self.client.search(**params)
        except Exception as e:
            logger.error(f"Search execution failed: {e}")
            return []
    
    def _delete_by_query(self, delete_query: dict) -> bool:
        """Delete documents matching query."""
        try:
            if hasattr(self.client, 'delete_by_query'):
                return self.client.delete_by_query(delete_query)
            else:
                logger.warning("delete_by_query method not available on OpenSearch client")
                return False
        except Exception as e:
            logger.error(f"Delete by query failed: {e}")
            return False
    
    def _build_aggregation_params(self, query: SearchQuery) -> dict:
        """Build aggregation parameters for faceted search."""
        return {
            "query": query.text,
            "aggs": {
                "doc_types": {"terms": {"field": "doc_type"}},
                "faculties": {"terms": {"field": "faculty"}},
                "years": {"terms": {"field": "year"}},
                "subjects": {"terms": {"field": "subject"}}
            }
        }
    
    def _execute_aggregation(self, params: dict) -> dict:
        """Execute aggregation query."""
        try:
            if hasattr(self.client, 'aggregate'):
                return self.client.aggregate(**params)
            else:
                logger.warning("aggregate method not available on OpenSearch client")
                return {}
        except Exception as e:
            logger.error(f"Aggregation execution failed: {e}")
            return {}
