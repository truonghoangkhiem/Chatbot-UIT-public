"""
Domain exceptions for the orchestration service.

These exceptions represent business rule violations and domain-specific errors.
They should NOT contain user-facing messages - those are handled by the presentation layer.
"""

from typing import Optional, Dict, Any


class OrchestrationDomainException(Exception):
    """Base domain exception for orchestration errors."""
    
    def __init__(self, error_code: str, details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        """
        Initialize domain exception.
        
        Args:
            error_code: Domain-specific error code (not user-facing)
            details: Technical details about the error
            cause: Original exception that caused this error
        """
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        super().__init__(error_code)


class AgentProcessingFailedException(OrchestrationDomainException):
    """Raised when agent processing fails."""
    
    def __init__(self, agent_error: str, details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        """
        Initialize agent processing failure.
        
        Args:
            agent_error: Technical description of the agent error
            details: Additional technical details
            cause: Original exception from agent service
        """
        super().__init__(
            error_code="AGENT_PROCESSING_FAILED",
            details={
                "agent_error": agent_error,
                **(details or {})
            },
            cause=cause
        )


class RAGRetrievalFailedException(OrchestrationDomainException):
    """Raised when RAG retrieval fails."""
    
    def __init__(self, rag_error: str, details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        """
        Initialize RAG retrieval failure.
        
        Args:
            rag_error: Technical description of the RAG error
            details: Additional technical details
            cause: Original exception from RAG service
        """
        super().__init__(
            error_code="RAG_RETRIEVAL_FAILED", 
            details={
                "rag_error": rag_error,
                **(details or {})
            },
            cause=cause
        )


class ContextManagementFailedException(OrchestrationDomainException):
    """Raised when conversation context management fails."""
    
    def __init__(self, context_error: str, details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        """
        Initialize context management failure.
        
        Args:
            context_error: Technical description of the context error
            details: Additional technical details
            cause: Original exception from context management
        """
        super().__init__(
            error_code="CONTEXT_MANAGEMENT_FAILED",
            details={
                "context_error": context_error,
                **(details or {})
            },
            cause=cause
        )