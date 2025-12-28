# app/api/schemas/doc.py
#
# Description:
# This file defines Pydantic models related to document management and indexing processes.

from pydantic import BaseModel
from typing import Optional, Dict, Any

class ManifestSpec(BaseModel):
    """
    Defines the specification for a document indexing manifest.
    This model is used as the request body for the re-indexing endpoint.
    """
    source_dir: Optional[str] = None
    glob: str = "**/*"
    metadata: Dict[str, Any] = {}
    chunk_size: int = 512
    chunk_overlap: int = 64

class ReindexResponse(BaseModel):
    """
    Defines the response structure for the re-indexing endpoint.
    """
    status: str
    indexed_docs: int