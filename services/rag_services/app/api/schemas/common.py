# app/api/schemas/common.py
#
# Description:
# This file defines common Pydantic models that are reused across multiple API endpoints.
# Centralizing these shared schemas promotes code reuse and consistency.

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class SourceMeta(BaseModel):
    """
    Defines the metadata structure for a retrieved document source.
    This provides context about where a piece of information originated.
    """
    doc_id: str
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

class CharacterSpan(BaseModel):
    """
    Defines a character span within a document for precise citation.
    """
    start: int
    end: int
    text: str
    type: str = "content"  # "content", "sentence", "paragraph"
    
class CitationSpan(BaseModel):
    """
    Enhanced citation structure with character-level precision.
    """
    doc_id: str
    chunk_id: Optional[str] = None
    page: Optional[int] = None
    char_spans: Optional[List[CharacterSpan]] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    highlighted_text: Optional[List[str]] = None

class Pagination(BaseModel):
    """
    Defines a standard pagination schema for list-based API responses.
    """
    offset: int = 0
    limit: int = 10