"""
Test cases for IRCoT (Interleaving Retrieval with Chain-of-Thought) implementation.

These tests verify that:
1. IRCoT configuration works correctly
2. IRCoT service performs iterative retrieval with reasoning
3. Integration with OrchestrationService and OptimizedOrchestrator works
4. Complex queries trigger IRCoT while simple queries bypass it
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

# Import IRCoT modules
from app.core.ircot_config import (
    IRCoTConfig, 
    IRCoTMode, 
    IRCoTIterationResult, 
    IRCoTResult,
    DEFAULT_IRCOT_CONFIG
)
from app.core.ircot_service import IRCoTReasoningService


class TestIRCoTConfig:
    """Test cases for IRCoT configuration."""
    
    def test_default_config_values(self):
        """Test that default config has expected values."""
        config = IRCoTConfig()
        
        assert config.enabled == True
        assert config.mode == IRCoTMode.AUTOMATIC
        assert config.max_iterations == 3
        assert config.min_iterations == 1
        assert config.complexity_threshold == 6.5
        assert config.context_accumulation == True
        assert config.retrieval_top_k == 3
        assert config.early_stopping_enabled == True
        assert config.early_stopping_threshold == 0.85
        
    def test_should_use_ircot_disabled(self):
        """Test that disabled IRCoT returns False."""
        config = IRCoTConfig(enabled=False)
        
        assert config.should_use_ircot("complex", 8.0) == False
        assert config.should_use_ircot("simple", 2.0) == False
        
    def test_should_use_ircot_forced_mode(self):
        """Test that forced mode always returns True (when enabled)."""
        config = IRCoTConfig(enabled=True, mode=IRCoTMode.FORCED)
        
        assert config.should_use_ircot("simple", 2.0) == True
        assert config.should_use_ircot("medium", 5.0) == True
        assert config.should_use_ircot("complex", 8.0) == True
        
    def test_should_use_ircot_automatic_mode_complex(self):
        """Test automatic mode triggers for complex queries."""
        config = IRCoTConfig(enabled=True, mode=IRCoTMode.AUTOMATIC)
        
        # Complex queries should trigger IRCoT
        assert config.should_use_ircot("complex", 8.0) == True
        assert config.should_use_ircot("complex", 7.0) == True
        
        # Score above threshold should trigger
        assert config.should_use_ircot("medium", 7.0) == True  # >= 6.5
        
    def test_should_use_ircot_automatic_mode_simple(self):
        """Test automatic mode does not trigger for simple queries."""
        config = IRCoTConfig(enabled=True, mode=IRCoTMode.AUTOMATIC)
        
        # Simple queries should not trigger IRCoT
        assert config.should_use_ircot("simple", 2.0) == False
        assert config.should_use_ircot("medium", 5.0) == False
        
    def test_config_to_dict(self):
        """Test config serialization."""
        config = IRCoTConfig()
        config_dict = config.to_dict()
        
        assert "enabled" in config_dict
        assert "mode" in config_dict
        assert "max_iterations" in config_dict
        assert config_dict["mode"] == "automatic"


class TestIRCoTIterationResult:
    """Test cases for IRCoT iteration results."""
    
    def test_iteration_result_to_dict(self):
        """Test iteration result serialization."""
        result = IRCoTIterationResult(
            iteration=1,
            reasoning_step="Phân tích thông tin về điều kiện đăng ký học phần",
            next_search_query="điều kiện tiên quyết môn học",
            retrieved_documents=[{"title": "Doc 1"}],
            confidence=0.6,
            can_answer_now=False,
            information_gaps=["Thông tin về điểm số tối thiểu"]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["iteration"] == 1
        assert result_dict["reasoning_step"] == "Phân tích thông tin về điều kiện đăng ký học phần"
        assert result_dict["documents_count"] == 1
        assert result_dict["confidence"] == 0.6
        assert result_dict["can_answer_now"] == False


class TestIRCoTResult:
    """Test cases for final IRCoT results."""
    
    def test_get_all_reasoning_steps(self):
        """Test extracting all reasoning steps."""
        iterations = [
            IRCoTIterationResult(
                iteration=1,
                reasoning_step="Step 1: Tìm thông tin cơ bản",
                next_search_query="query 1",
                retrieved_documents=[],
                confidence=0.4,
                can_answer_now=False
            ),
            IRCoTIterationResult(
                iteration=2,
                reasoning_step="Step 2: Phân tích chi tiết",
                next_search_query=None,
                retrieved_documents=[],
                confidence=0.8,
                can_answer_now=True
            )
        ]
        
        result = IRCoTResult(
            original_query="Điều kiện đăng ký học phần SE101",
            iterations=iterations,
            total_iterations=2,
            accumulated_context=[],
            final_reasoning="Combined reasoning",
            final_confidence=0.8
        )
        
        steps = result.get_all_reasoning_steps()
        assert len(steps) == 2
        assert "Step 1" in steps[0]
        assert "Step 2" in steps[1]
        
    def test_get_all_search_queries(self):
        """Test extracting all search queries."""
        iterations = [
            IRCoTIterationResult(
                iteration=1,
                reasoning_step="Step 1",
                next_search_query="điều kiện tiên quyết",
                retrieved_documents=[],
                confidence=0.4,
                can_answer_now=False
            ),
            IRCoTIterationResult(
                iteration=2,
                reasoning_step="Step 2",
                next_search_query="điểm số yêu cầu",
                retrieved_documents=[],
                confidence=0.8,
                can_answer_now=True
            )
        ]
        
        result = IRCoTResult(
            original_query="Điều kiện đăng ký học phần",
            iterations=iterations,
            total_iterations=2,
            accumulated_context=[],
            final_reasoning="",
            final_confidence=0.8
        )
        
        queries = result.get_all_search_queries()
        assert len(queries) == 3  # original + 2 from iterations
        assert queries[0] == "Điều kiện đăng ký học phần"
        assert "điều kiện tiên quyết" in queries


class TestIRCoTService:
    """Test cases for IRCoT reasoning service."""
    
    @pytest.fixture
    def mock_agent_port(self):
        """Create mock agent port."""
        agent_port = AsyncMock()
        agent_port.generate_response = AsyncMock(return_value=MagicMock(
            content='{"reasoning_step": "Analyzing query", "information_gaps": [], "next_search_query": null, "confidence": 0.9, "can_answer_now": true}'
        ))
        return agent_port
    
    @pytest.fixture
    def mock_rag_port(self):
        """Create mock RAG port."""
        rag_port = AsyncMock()
        rag_port.retrieve_context = AsyncMock(return_value={
            "retrieved_documents": [
                {"text": "Document content 1", "title": "Doc 1", "score": 0.9},
                {"text": "Document content 2", "title": "Doc 2", "score": 0.8}
            ]
        })
        return rag_port
    
    @pytest.fixture
    def ircot_service(self, mock_agent_port, mock_rag_port):
        """Create IRCoT service with mocks."""
        config = IRCoTConfig(
            enabled=True,
            mode=IRCoTMode.FORCED,
            max_iterations=2
        )
        return IRCoTReasoningService(
            agent_port=mock_agent_port,
            rag_port=mock_rag_port,
            config=config
        )
    
    @pytest.mark.asyncio
    async def test_reason_with_retrieval_single_iteration(self, ircot_service, mock_rag_port):
        """Test IRCoT with single iteration (high confidence)."""
        result = await ircot_service.reason_with_retrieval(
            query="Điều kiện đăng ký học phần SE101?",
            initial_context=None
        )
        
        assert result is not None
        assert result.original_query == "Điều kiện đăng ký học phần SE101?"
        assert result.total_iterations >= 1
        assert len(result.accumulated_context) > 0
        
    @pytest.mark.asyncio
    async def test_reason_with_retrieval_accumulates_context(self, mock_agent_port, mock_rag_port):
        """Test that IRCoT accumulates context across iterations."""
        # Setup multi-iteration responses
        responses = [
            MagicMock(content='{"reasoning_step": "Step 1", "information_gaps": ["more info"], "next_search_query": "follow up query", "confidence": 0.5, "can_answer_now": false}'),
            MagicMock(content='{"reasoning_step": "Step 2", "information_gaps": [], "next_search_query": null, "confidence": 0.9, "can_answer_now": true}')
        ]
        mock_agent_port.generate_response = AsyncMock(side_effect=responses)
        
        config = IRCoTConfig(enabled=True, max_iterations=3, min_iterations=1)
        service = IRCoTReasoningService(mock_agent_port, mock_rag_port, config)
        
        result = await service.reason_with_retrieval("Complex multi-hop question")
        
        # Should have 2 iterations
        assert result.total_iterations == 2
        # Should have accumulated documents from both iterations
        assert len(result.accumulated_context) > 0
        
    @pytest.mark.asyncio
    async def test_early_stopping(self, mock_agent_port, mock_rag_port):
        """Test that early stopping works when confidence is high."""
        # High confidence response
        mock_agent_port.generate_response = AsyncMock(return_value=MagicMock(
            content='{"reasoning_step": "Complete answer", "information_gaps": [], "next_search_query": null, "confidence": 0.95, "can_answer_now": true}'
        ))
        
        config = IRCoTConfig(
            enabled=True, 
            max_iterations=5,
            early_stopping_enabled=True,
            early_stopping_threshold=0.85
        )
        service = IRCoTReasoningService(mock_agent_port, mock_rag_port, config)
        
        result = await service.reason_with_retrieval("Simple question")
        
        # Should stop early (not use all 5 iterations)
        assert result.total_iterations < 5
        assert result.early_stopped == True
        
    @pytest.mark.asyncio
    async def test_deduplication(self, mock_agent_port, mock_rag_port):
        """Test that duplicate documents are removed."""
        # Return same documents on multiple calls
        mock_rag_port.retrieve_context = AsyncMock(return_value={
            "retrieved_documents": [
                {"text": "Same content here", "title": "Doc 1", "score": 0.9},
                {"text": "Same content here", "title": "Doc 1 copy", "score": 0.85}
            ]
        })
        
        config = IRCoTConfig(enabled=True, deduplication_enabled=True, max_iterations=2)
        service = IRCoTReasoningService(mock_agent_port, mock_rag_port, config)
        
        result = await service.reason_with_retrieval("Query")
        
        # Should have deduplicated documents
        unique_contents = set()
        for doc in result.accumulated_context:
            content = doc.get("text", doc.get("content", ""))
            unique_contents.add(content[:100])
        
        # All accumulated documents should have unique content prefixes
        assert len(unique_contents) == len(result.accumulated_context)


class TestIRCoTIntegration:
    """Integration tests for IRCoT with orchestration services."""
    
    @pytest.mark.asyncio
    async def test_orchestration_service_with_ircot(self):
        """Test OrchestrationService uses IRCoT for complex queries."""
        from app.core.orchestration_service import OrchestrationService
        from app.core.domain import OrchestrationRequest
        
        # Create mocks
        mock_agent_port = AsyncMock()
        mock_agent_port.generate_response = AsyncMock(return_value=MagicMock(
            content="Response content",
            tokens_used=100,
            metadata={}
        ))
        mock_agent_port.validate_connection = AsyncMock(return_value=True)
        
        mock_rag_port = AsyncMock()
        mock_rag_port.retrieve_context = AsyncMock(return_value={
            "retrieved_documents": [{"text": "Doc", "score": 0.9}]
        })
        mock_rag_port.health_check = AsyncMock(return_value=True)
        
        mock_conversation_manager = AsyncMock()
        mock_conversation_manager.get_context = AsyncMock(return_value=None)
        mock_conversation_manager.create_context = AsyncMock(return_value=MagicMock(
            add_message=MagicMock(),
            metadata={}
        ))
        mock_conversation_manager.update_context = AsyncMock()
        
        # Create service with IRCoT enabled
        ircot_config = IRCoTConfig(enabled=True, mode=IRCoTMode.AUTOMATIC)
        service = OrchestrationService(
            agent_port=mock_agent_port,
            rag_port=mock_rag_port,
            conversation_manager=mock_conversation_manager,
            ircot_config=ircot_config
        )
        
        # Test with complex query
        request = OrchestrationRequest(
            user_query="Nếu tôi không đạt điều kiện tiên quyết thì có thể đăng ký học phần SE101 không?",
            use_rag=True
        )
        
        # Call the IRCoT method
        response = await service.process_request_with_ircot(
            request, 
            complexity="complex", 
            complexity_score=7.5
        )
        
        assert response is not None
        assert "ircot" in response.processing_stats.get("ircot_mode", False) or response.processing_stats.get("ircot_mode", False)


# Example complex queries for manual testing
COMPLEX_QUERY_EXAMPLES = [
    "Nếu tôi không đạt điều kiện tiên quyết môn SE101 thì những môn học nào sẽ bị ảnh hưởng theo dây chuyền?",
    "So sánh chương trình đào tạo ngành CNTT và ngành HTTT về số tín chỉ và môn học bắt buộc?",
    "Điều kiện để được xét tốt nghiệp sớm là gì và cần chuẩn bị những gì?",
    "Nếu điểm trung bình chung học kỳ dưới 5.0 thì sinh viên cần làm gì để tránh bị cảnh báo học vụ?",
]

SIMPLE_QUERY_EXAMPLES = [
    "Điểm tối đa của một môn học là bao nhiêu?",
    "Địa chỉ trường UIT ở đâu?",
    "Thời gian nghỉ hè là khi nào?",
]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
