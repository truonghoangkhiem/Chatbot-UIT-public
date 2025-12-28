# app/api/v1/routes/admin.py
#
# Description:
# This module implements administrative endpoints for the RAG service.
# It currently includes the endpoint for triggering the document re-indexing process.

from fastapi import APIRouter, HTTPException
from app.api.schemas.doc import ManifestSpec, ReindexResponse
# NOTE: Old indexing module removed - use scripts/index_semantic_data.py instead
# from indexing.pipeline import reindex_from_manifest

# Create an API router for admin-related endpoints
router = APIRouter(tags=["admin"])

@router.post("/admin/reindex", response_model=ReindexResponse)
def reindex(manifest: ManifestSpec):
    """
    Endpoint to trigger the re-indexing of documents from a specified source directory.
    
    NOTE: This endpoint is deprecated. Use scripts/index_semantic_data.py for indexing.

    Args:
        manifest (ManifestSpec): The manifest detailing the source and configuration for indexing.

    Returns:
        ReindexResponse: The status and count of indexed documents.
    """
    raise HTTPException(
        status_code=501, 
        detail="This endpoint is deprecated. Use scripts/index_semantic_data.py for indexing."
    )