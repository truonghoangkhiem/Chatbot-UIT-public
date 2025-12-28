"""
Exception handlers for converting domain exceptions to user-friendly API responses.

This module handles the conversion from domain exceptions (which contain technical details)
to presentation-layer responses with appropriate user messages.
"""

from typing import Dict, Any
from ..core.exceptions import (
    OrchestrationDomainException,
    AgentProcessingFailedException,
    RAGRetrievalFailedException,
    ContextManagementFailedException
)


class ExceptionMessageHandler:
    """Handles conversion of domain exceptions to user-friendly messages."""
    
    @staticmethod
    def get_user_message(exception: OrchestrationDomainException) -> str:
        """
        Convert domain exception to user-friendly message.
        
        Args:
            exception: Domain exception from core layer
            
        Returns:
            User-friendly error message in Vietnamese
        """
        if isinstance(exception, AgentProcessingFailedException):
            return "Xin lỗi, đã có lỗi xảy ra khi xử lý yêu cầu của bạn. Vui lòng thử lại sau."
        
        elif isinstance(exception, RAGRetrievalFailedException):
            return "Không thể tìm kiếm thông tin từ cơ sở dữ liệu. Vui lòng thử lại hoặc liên hệ hỗ trợ."
        
        elif isinstance(exception, ContextManagementFailedException):
            return "Đã có lỗi với phiên trò chuyện. Vui lòng bắt đầu cuộc trò chuyện mới."
        
        else:
            # Generic domain exception
            return "Đã có lỗi hệ thống. Vui lòng thử lại sau hoặc liên hệ hỗ trợ kỹ thuật."
    
    @staticmethod
    def get_error_details(exception: OrchestrationDomainException) -> Dict[str, Any]:
        """
        Extract technical details for logging/debugging.
        
        Args:
            exception: Domain exception from core layer
            
        Returns:
            Technical details for internal use
        """
        return {
            "error_code": exception.error_code,
            "details": exception.details,
            "cause": str(exception.cause) if exception.cause else None,
            "exception_type": type(exception).__name__
        }
    
    @staticmethod 
    def get_http_status_code(exception: OrchestrationDomainException) -> int:
        """
        Get appropriate HTTP status code for domain exception.
        
        Args:
            exception: Domain exception from core layer
            
        Returns:
            HTTP status code
        """
        if isinstance(exception, AgentProcessingFailedException):
            return 503  # Service Unavailable
        
        elif isinstance(exception, RAGRetrievalFailedException):
            return 503  # Service Unavailable 
        
        elif isinstance(exception, ContextManagementFailedException):
            return 500  # Internal Server Error
        
        else:
            return 500  # Internal Server Error
    
    @staticmethod
    def create_fallback_response(
        exception: OrchestrationDomainException,
        session_id: str,
        user_query: str
    ) -> Dict[str, Any]:
        """
        Create a fallback response when agent processing fails.
        
        Args:
            exception: Domain exception that occurred
            session_id: Session ID for the request
            user_query: Original user query
            
        Returns:
            Fallback response structure
        """
        user_message = ExceptionMessageHandler.get_user_message(exception)
        
        return {
            "response": user_message,
            "session_id": session_id,
            "is_error_response": True,
            "error_handled": True,
            "original_query": user_query,
            "suggested_actions": [
                "Thử lại yêu cầu sau vài phút",
                "Đơn giản hoá câu hỏi của bạn", 
                "Liên hệ hỗ trợ kỹ thuật nếu lỗi tiếp tục xảy ra"
            ]
        }