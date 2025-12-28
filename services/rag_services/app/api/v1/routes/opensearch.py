# app/api/v1/routes/opensearch.py
#
# Description:
# API endpoints for managing OpenSearch indexing and BM25 search functionality.
# Provides endpoints for document indexing, search testing, and index management.

import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.api.schemas.common import SourceMeta
from app.config.settings import settings

# Import OpenSearch client with error handling
try:
    from infrastructure.store.opensearch.client import get_opensearch_client
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(tags=["opensearch"])

# --- Request/Response Models ---

class IndexDocumentRequest(BaseModel):
    """Enhanced request model for indexing a document with Vietnamese support."""
    doc_id: str
    chunk_id: str
    text: str
    title: Optional[str] = None
    doc_type: Optional[str] = "document"
    faculty: Optional[str] = "general"
    year: Optional[int] = None
    subject: Optional[str] = None
    language: str = "vi"
    metadata: Optional[Dict[str, Any]] = None

class BulkIndexRequest(BaseModel):
    """Request model for bulk indexing documents."""
    documents: List[IndexDocumentRequest]

class BM25SearchRequest(BaseModel):
    """Enhanced request model for BM25-only search with Vietnamese support."""
    query: str
    size: int = 10
    filters: Optional[Dict[str, Any]] = None
    # Field-specific filters
    doc_types: Optional[List[str]] = None
    faculties: Optional[List[str]] = None
    years: Optional[List[int]] = None
    subjects: Optional[List[str]] = None
    language: Optional[str] = "vi"
    include_char_spans: bool = True
    highlight_matches: bool = True

class BM25SearchHit(BaseModel):
    """Enhanced response model for BM25 search results."""
    doc_id: str
    chunk_id: str
    text: str
    title: Optional[str] = None
    doc_type: Optional[str] = None
    faculty: Optional[str] = None
    year: Optional[int] = None
    subject: Optional[str] = None
    language: Optional[str] = None
    metadata: Dict[str, Any]
    bm25_score: float
    highlighted_text: Optional[List[str]] = None
    highlighted_title: Optional[List[str]] = None
    char_spans: Optional[List[Dict[str, Any]]] = None

class BM25SearchResponse(BaseModel):
    """Response model for BM25 search."""
    hits: List[BM25SearchHit]
    total_hits: int
    latency_ms: int

class IndexStatsResponse(BaseModel):
    """Response model for index statistics."""
    total_documents: int
    index_size_bytes: int
    opensearch_available: bool
    index_name: str

# --- Endpoints ---

@router.get("/opensearch/health")
def opensearch_health():
    """Check OpenSearch health and connectivity."""
    if not OPENSEARCH_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "OpenSearch dependencies not installed",
            "available": False
        }
    
    try:
        client = get_opensearch_client()
        is_healthy = client.health_check()
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "available": True,
            "opensearch_host": f"{settings.opensearch_host}:{settings.opensearch_port}",
            "index_name": settings.opensearch_index
        }
    except Exception as e:
        logger.error(f"OpenSearch health check failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "available": False
        }

@router.post("/opensearch/index-document")
def index_document(request: IndexDocumentRequest):
    """Index a single document chunk with enhanced metadata in OpenSearch."""
    if not OPENSEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenSearch not available")
    
    try:
        client = get_opensearch_client()
        success = client.index_document(
            doc_id=request.doc_id,
            chunk_id=request.chunk_id,
            text=request.text,
            title=request.title,
            doc_type=request.doc_type,
            faculty=request.faculty,
            year=request.year,
            subject=request.subject,
            language=request.language,
            metadata=request.metadata
        )
        
        if success:
            return {"message": "Document indexed successfully", "success": True}
        else:
            raise HTTPException(status_code=500, detail="Failed to index document")
            
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error indexing document: {str(e)}")

@router.post("/opensearch/bulk-index")
def bulk_index_documents(request: BulkIndexRequest, background_tasks: BackgroundTasks):
    """Bulk index multiple documents in OpenSearch."""
    if not OPENSEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenSearch not available")
    
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")
    
    try:
        client = get_opensearch_client()
        
        # Convert request documents to the format expected by the client
        documents = []
        for doc in request.documents:
            documents.append({
                "doc_id": doc.doc_id,
                "chunk_id": doc.chunk_id,
                "text": doc.text,
                "metadata": doc.metadata or {}
            })
        
        success_count, failed_count = client.bulk_index_documents(documents)
        
        return {
            "message": f"Bulk indexing completed",
            "total_documents": len(request.documents),
            "successful": success_count,
            "failed": failed_count,
            "success": failed_count == 0
        }
        
    except Exception as e:
        logger.error(f"Error in bulk indexing: {e}")
        raise HTTPException(status_code=500, detail=f"Error in bulk indexing: {str(e)}")

@router.post("/opensearch/search", response_model=BM25SearchResponse)
def bm25_search(request: BM25SearchRequest):
    """Perform BM25-only search using OpenSearch."""
    if not OPENSEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenSearch not available")
    
    import time
    start_time = time.time()
    
    try:
        client = get_opensearch_client()
        results = client.search(
            query=request.query,
            size=request.size,
            filters=request.filters,
            doc_types=request.doc_types,
            faculties=request.faculties,
            years=request.years,
            subjects=request.subjects,
            language=request.language,
            include_char_spans=request.include_char_spans
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Convert results to response format
        hits = []
        for result in results:
            hit = BM25SearchHit(
                doc_id=result["doc_id"],
                chunk_id=result["chunk_id"],
                text=result["text"],
                title=result.get("title", ""),
                doc_type=result.get("doc_type"),
                faculty=result.get("faculty"),
                year=result.get("year"),
                subject=result.get("subject"),
                language=result.get("language"),
                metadata=result["metadata"],
                bm25_score=result["bm25_score"],
                highlighted_text=result.get("highlighted_text"),
                highlighted_title=result.get("highlighted_title"),
                char_spans=result.get("char_spans")
            )
            hits.append(hit)
        
        return BM25SearchResponse(
            hits=hits,
            total_hits=len(hits),
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"Error in BM25 search: {e}")
        raise HTTPException(status_code=500, detail=f"Error in BM25 search: {str(e)}")

@router.get("/opensearch/stats", response_model=IndexStatsResponse)
def get_index_stats():
    """Get OpenSearch index statistics."""
    if not OPENSEARCH_AVAILABLE:
        return IndexStatsResponse(
            total_documents=0,
            index_size_bytes=0,
            opensearch_available=False,
            index_name=settings.opensearch_index
        )
    
    try:
        client = get_opensearch_client()
        stats = client.get_index_stats()
        
        return IndexStatsResponse(
            total_documents=stats["total_docs"],
            index_size_bytes=stats["index_size"],
            opensearch_available=True,
            index_name=settings.opensearch_index
        )
        
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        return IndexStatsResponse(
            total_documents=0,
            index_size_bytes=0,
            opensearch_available=False,
            index_name=settings.opensearch_index
        )

@router.delete("/opensearch/document/{doc_id}")
def delete_document_by_id(doc_id: str):
    """Delete all chunks for a given document ID."""
    if not OPENSEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenSearch not available")
    
    try:
        client = get_opensearch_client()
        deleted_count = client.delete_all_documents_for_doc_id(doc_id)
        
        return {
            "message": f"Deleted {deleted_count} chunks for document {doc_id}",
            "deleted_count": deleted_count,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@router.delete("/opensearch/document/{doc_id}/{chunk_id}")
def delete_document_chunk(doc_id: str, chunk_id: str):
    """Delete a specific document chunk."""
    if not OPENSEARCH_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenSearch not available")
    
    try:
        client = get_opensearch_client()
        success = client.delete_document(doc_id, chunk_id)
        
        if success:
            return {
                "message": f"Deleted chunk {chunk_id} from document {doc_id}",
                "success": True
            }
        else:
            raise HTTPException(status_code=404, detail="Document chunk not found")
            
    except Exception as e:
        logger.error(f"Error deleting document chunk {doc_id}/{chunk_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document chunk: {str(e)}")
