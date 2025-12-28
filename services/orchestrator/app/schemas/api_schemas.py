"""
API schemas for orchestrator service.

This module defines the request and response schemas for the orchestrator API,
providing clear contracts for client-server communication.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    
    query: str = Field(..., description="User query/message", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    use_rag: bool = Field(True, description="Whether to use RAG for context retrieval")
    use_knowledge_graph: Optional[bool] = Field(None, description="Whether to force Knowledge Graph usage (overrides SmartPlanner)")
    rag_top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    model: Optional[str] = Field(None, description="Specific model to use for generation")
    temperature: Optional[float] = Field(None, description="Generation temperature", ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate", ge=1, le=4000)
    stream: bool = Field(False, description="Whether to stream the response")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Hướng dẫn đăng ký học phần tại UIT như thế nào?",
                "session_id": "user_123_session",
                "use_rag": True,
                "rag_top_k": 5,
                "model": "openai/gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000,
                "stream": False
            }
        }


class DocumentInfo(BaseModel):
    """Schema for document information in RAG context."""
    
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    score: float = Field(..., description="Relevance score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional document metadata")


class RAGContextInfo(BaseModel):
    """Schema for RAG context information."""
    
    query: str = Field(..., description="Original search query")
    documents: List[DocumentInfo] = Field(..., description="Retrieved documents")
    total_documents: int = Field(..., description="Total number of documents retrieved")
    search_mode: str = Field(..., description="Search mode used")
    processing_time: Optional[float] = Field(None, description="RAG processing time in seconds")
    
    # Search source indicators
    use_knowledge_graph: bool = Field(False, description="Whether Knowledge Graph was used")
    use_vector_search: bool = Field(True, description="Whether Vector Search was used")
    
    # Smart planner info
    complexity: Optional[str] = Field(None, description="Query complexity: simple/medium/complex")
    strategy: Optional[str] = Field(None, description="RAG strategy used")


class ProcessingStats(BaseModel):
    """Schema for processing statistics."""
    
    total_time: float = Field(..., description="Total processing time in seconds")
    rag_time: Optional[float] = Field(None, description="RAG processing time in seconds")
    agent_time: Optional[float] = Field(None, description="Agent processing time in seconds")
    documents_retrieved: Optional[int] = Field(None, description="Number of documents retrieved")
    tokens_used: Optional[int] = Field(None, description="Number of tokens used")
    rag_error: Optional[str] = Field(None, description="RAG error message if any")
    
    # Optimized pipeline stats
    llm_calls: Optional[int] = Field(None, description="Number of LLM API calls made")
    pipeline: Optional[str] = Field(None, description="Pipeline type used")
    planning_time: Optional[float] = Field(None, description="Planning step time in seconds")
    answer_generation_time: Optional[float] = Field(None, description="Answer generation time in seconds")
    plan_complexity: Optional[str] = Field(None, description="Query complexity from planner")
    plan_complexity_score: Optional[float] = Field(None, description="Numeric complexity score")


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    
    response: str = Field(..., description="Generated response")
    session_id: str = Field(..., description="Conversation session ID")
    timestamp: datetime = Field(..., description="Response timestamp")
    rag_context: Optional[RAGContextInfo] = Field(None, description="RAG context used")
    processing_stats: ProcessingStats = Field(..., description="Processing statistics")
    model_used: Optional[str] = Field(None, description="Model used for generation")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "Để đăng ký học phần tại UIT, bạn cần làm theo các bước sau...",
                "session_id": "user_123_session",
                "timestamp": "2024-01-01T12:00:00Z",
                "rag_context": {
                    "query": "Hướng dẫn đăng ký học phần tại UIT như thế nào?",
                    "documents": [
                        {
                            "title": "Hướng dẫn đăng ký học phần",
                            "content": "Quy trình đăng ký học phần...",
                            "score": 0.85,
                            "metadata": {"source": "handbook"}
                        }
                    ],
                    "total_documents": 3,
                    "search_mode": "hybrid",
                    "processing_time": 0.5
                },
                "processing_stats": {
                    "total_time": 2.1,
                    "rag_time": 0.5,
                    "agent_time": 1.4,
                    "documents_retrieved": 3,
                    "tokens_used": 250
                },
                "model_used": "openai/gpt-3.5-turbo"
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    services: Dict[str, Any] = Field(..., description="Status of individual services")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "services": {
                    "orchestrator": "healthy",
                    "agent": "healthy",
                    "rag": "healthy",
                    "conversation_manager": "healthy",
                    "ircot_enabled": True
                }
            }
        }


class ConversationInfo(BaseModel):
    """Schema for conversation information."""
    
    session_id: str = Field(..., description="Session ID")
    message_count: int = Field(..., description="Number of messages in conversation")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


class ConversationsResponse(BaseModel):
    """Response schema for conversations endpoint."""
    
    conversations: List[ConversationInfo] = Field(..., description="List of active conversations")
    total_count: int = Field(..., description="Total number of active conversations")
    
    class Config:
        schema_extra = {
            "example": {
                "conversations": [
                    {
                        "session_id": "user_123_session",
                        "message_count": 6,
                        "created_at": "2024-01-01T10:00:00Z",
                        "updated_at": "2024-01-01T12:00:00Z"
                    }
                ],
                "total_count": 1
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for error responses."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(..., description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Invalid request",
                "detail": "Query field is required",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }