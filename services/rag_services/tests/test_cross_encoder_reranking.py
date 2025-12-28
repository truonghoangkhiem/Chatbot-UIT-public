"""
Tests for cross-encoder reranking functionality.

This module tests the reranking service implementation to ensure
proper integration and functionality within the RAG system.
"""

import pytest
import asyncio
from typing import List
from unittest.mock import Mock, patch

from core.domain.models import (
    SearchResult, DocumentMetadata, DocumentLanguage, RerankingMetadata
)
from adapters.cross_encoder_reranker import (
    CrossEncoderRerankingService,
    MultilingualCrossEncoderRerankingService,
    NoOpRerankingService,
    create_reranking_service
)


class TestCrossEncoderRerankingService:
    """Test cases for CrossEncoderRerankingService."""
    
    @pytest.fixture
    def sample_search_results(self) -> List[SearchResult]:
        """Create sample search results for testing."""
        return [
            SearchResult(
                text="Quy định về học phí tại Đại học Công nghệ Thông tin",
                metadata=DocumentMetadata(
                    doc_id="doc1",
                    chunk_id="chunk1",
                    title="Quy định học phí",
                    doc_type="regulation",
                    faculty="CNTT"
                ),
                score=0.8,
                source_type="vector"
            ),
            SearchResult(
                text="Hướng dẫn đăng ký học phần cho sinh viên năm nhất",
                metadata=DocumentMetadata(
                    doc_id="doc2",
                    chunk_id="chunk2",
                    title="Hướng dẫn đăng ký",
                    doc_type="guide",
                    faculty="CNTT"
                ),
                score=0.7,
                source_type="vector"
            ),
            SearchResult(
                text="Chương trình đào tạo ngành Khoa học máy tính",
                metadata=DocumentMetadata(
                    doc_id="doc3",
                    chunk_id="chunk3",
                    title="Chương trình đào tạo",
                    doc_type="curriculum",
                    faculty="CNTT"
                ),
                score=0.6,
                source_type="vector"
            )
        ]
    
    @pytest.fixture
    def mock_cross_encoder(self):
        """Create a mock CrossEncoder model."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.5, 0.8]  # Mock scores
        mock_model.device = "cpu"
        return mock_model
    
    @patch('adapters.cross_encoder_reranker.CrossEncoder')
    def test_initialization_success(self, mock_cross_encoder_class):
        """Test successful initialization of reranking service."""
        mock_cross_encoder_class.return_value = self.mock_cross_encoder()
        
        service = CrossEncoderRerankingService(
            model_name="test-model",
            batch_size=8,
            max_length=256
        )
        
        assert service.is_available()
        assert service.model_name == "test-model"
        assert service.batch_size == 8
        assert service.max_length == 256
        
        model_info = service.get_model_info()
        assert model_info["model_name"] == "test-model"
        assert model_info["batch_size"] == 8
    
    def test_initialization_failure(self):
        """Test initialization failure when sentence-transformers is not available."""
        with patch('adapters.cross_encoder_reranker.CrossEncoder', side_effect=ImportError("No module")):
            with pytest.raises(ImportError):
                CrossEncoderRerankingService("test-model")
    
    @patch('adapters.cross_encoder_reranker.CrossEncoder')
    @pytest.mark.asyncio
    async def test_rerank_success(self, mock_cross_encoder_class, sample_search_results):
        """Test successful reranking of search results."""
        mock_model = self.mock_cross_encoder()
        mock_cross_encoder_class.return_value = mock_model
        
        service = CrossEncoderRerankingService("test-model")
        
        query = "học phí đại học"
        reranked_results = await service.rerank(query, sample_search_results)
        
        # Verify results are reranked
        assert len(reranked_results) == len(sample_search_results)
        
        # Check if results are sorted by rerank score
        scores = [result.rerank_score for result in reranked_results]
        assert scores == sorted(scores, reverse=True)
        
        # Verify reranking metadata is added
        for i, result in enumerate(reranked_results):
            assert result.reranking_metadata is not None
            assert result.reranking_metadata.original_rank == sample_search_results.index(
                next(r for r in sample_search_results if r.text == result.text)
            ) + 1
            assert result.reranking_metadata.model_name == "test-model"
            assert result.rank == i + 1
    
    @patch('adapters.cross_encoder_reranker.CrossEncoder')
    @pytest.mark.asyncio
    async def test_rerank_with_top_k(self, mock_cross_encoder_class, sample_search_results):
        """Test reranking with top_k parameter."""
        mock_model = self.mock_cross_encoder()
        mock_cross_encoder_class.return_value = mock_model
        
        service = CrossEncoderRerankingService("test-model")
        
        query = "học phí đại học"
        reranked_results = await service.rerank(query, sample_search_results, top_k=2)
        
        # Should return only top 2 results
        assert len(reranked_results) == 2
        
        # Results should be sorted by rerank score
        assert reranked_results[0].rerank_score >= reranked_results[1].rerank_score
    
    @patch('adapters.cross_encoder_reranker.CrossEncoder')
    @pytest.mark.asyncio
    async def test_compute_relevance_scores(self, mock_cross_encoder_class):
        """Test computing relevance scores for query-text pairs."""
        mock_model = self.mock_cross_encoder()
        mock_cross_encoder_class.return_value = mock_model
        
        service = CrossEncoderRerankingService("test-model")
        
        query = "học phí"
        texts = ["học phí đại học", "đăng ký học phần", "chương trình đào tạo"]
        
        scores = await service.compute_relevance_scores(query, texts)
        
        assert len(scores) == len(texts)
        assert all(isinstance(score, float) for score in scores)
        
        # Verify the model was called with correct pairs
        expected_pairs = [(query, text) for text in texts]
        mock_model.predict.assert_called_once_with(expected_pairs)
    
    @pytest.mark.asyncio
    async def test_rerank_empty_results(self):
        """Test reranking with empty results list."""
        service = NoOpRerankingService()  # Use no-op service for this test
        
        query = "test query"
        empty_results = []
        
        reranked_results = await service.rerank(query, empty_results)
        
        assert reranked_results == []
    
    @patch('adapters.cross_encoder_reranker.CrossEncoder')
    @pytest.mark.asyncio
    async def test_rerank_model_error(self, mock_cross_encoder_class, sample_search_results):
        """Test reranking when model prediction fails."""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model error")
        mock_model.device = "cpu"
        mock_cross_encoder_class.return_value = mock_model
        
        service = CrossEncoderRerankingService("test-model")
        
        query = "test query"
        reranked_results = await service.rerank(query, sample_search_results)
        
        # Should return original results on error
        assert reranked_results == sample_search_results


class TestMultilingualCrossEncoderRerankingService:
    """Test cases for MultilingualCrossEncoderRerankingService."""
    
    def test_language_detection(self):
        """Test Vietnamese language detection."""
        service = MultilingualCrossEncoderRerankingService(auto_detect_language=True)
        
        # Vietnamese text
        vietnamese_text = "Quy định về học phí tại Đại học Công nghệ Thông tin"
        assert service._detect_language(vietnamese_text) == "vi"
        
        # English text
        english_text = "University tuition fee regulations"
        assert service._detect_language(english_text) == "en"
        
        # Mixed text with Vietnamese characters
        mixed_text = "Học phí and tuition fees"
        assert service._detect_language(mixed_text) == "vi"
    
    @patch('adapters.cross_encoder_reranker.CrossEncoder')
    def test_model_selection_vietnamese(self, mock_cross_encoder_class):
        """Test model selection for Vietnamese content."""
        # Create mock models
        default_model = Mock()
        default_model.device = "cpu"
        vietnamese_model = Mock()
        vietnamese_model.device = "cpu"
        
        mock_cross_encoder_class.side_effect = [default_model, vietnamese_model]
        
        service = MultilingualCrossEncoderRerankingService(
            model_name="default-model",
            vietnamese_model_name="vi-model"
        )
        
        # Test with Vietnamese content
        query = "học phí đại học"
        texts = ["Quy định học phí", "Hướng dẫn đăng ký"]
        
        selected_model = service._select_model(query, texts)
        assert selected_model == vietnamese_model  # Should select Vietnamese model
    
    @patch('adapters.cross_encoder_reranker.CrossEncoder')
    def test_model_selection_english(self, mock_cross_encoder_class):
        """Test model selection for English content."""
        # Create mock models
        default_model = Mock()
        default_model.device = "cpu"
        vietnamese_model = Mock()
        vietnamese_model.device = "cpu"
        
        mock_cross_encoder_class.side_effect = [default_model, vietnamese_model]
        
        service = MultilingualCrossEncoderRerankingService(
            model_name="default-model",
            vietnamese_model_name="vi-model"
        )
        
        # Test with English content
        query = "university tuition"
        texts = ["Tuition regulations", "Registration guide"]
        
        selected_model = service._select_model(query, texts)
        assert selected_model == default_model  # Should select default model


class TestRerankingServiceFactory:
    """Test cases for reranking service factory function."""
    
    @patch('adapters.cross_encoder_reranker.MultilingualCrossEncoderRerankingService')
    def test_create_multilingual_service(self, mock_multilingual_service):
        """Test creating multilingual reranking service."""
        mock_service = Mock()
        mock_multilingual_service.return_value = mock_service
        
        service = create_reranking_service(
            model_name="test-model",
            multilingual=True,
            batch_size=8
        )
        
        assert service == mock_service
        mock_multilingual_service.assert_called_once_with(
            model_name="test-model",
            batch_size=8
        )
    
    @patch('adapters.cross_encoder_reranker.CrossEncoderRerankingService')
    def test_create_standard_service(self, mock_standard_service):
        """Test creating standard reranking service."""
        mock_service = Mock()
        mock_standard_service.return_value = mock_service
        
        service = create_reranking_service(
            model_name="test-model",
            multilingual=False,
            max_length=256
        )
        
        assert service == mock_service
        mock_standard_service.assert_called_once_with(
            model_name="test-model",
            max_length=256
        )
    
    @patch('adapters.cross_encoder_reranker.MultilingualCrossEncoderRerankingService')
    def test_create_service_fallback_on_error(self, mock_multilingual_service):
        """Test fallback to no-op service when creation fails."""
        mock_multilingual_service.side_effect = Exception("Creation failed")
        
        service = create_reranking_service(model_name="test-model")
        
        # Should return NoOpRerankingService
        assert isinstance(service, NoOpRerankingService)
        assert service.is_available()


class TestNoOpRerankingService:
    """Test cases for NoOpRerankingService."""
    
    @pytest.fixture
    def service(self):
        """Create NoOpRerankingService instance."""
        return NoOpRerankingService()
    
    @pytest.mark.asyncio
    async def test_rerank_returns_unchanged(self, service):
        """Test that rerank returns results unchanged."""
        results = [Mock(), Mock(), Mock()]
        query = "test query"
        
        reranked_results = await service.rerank(query, results)
        
        assert reranked_results == results
    
    @pytest.mark.asyncio
    async def test_rerank_with_top_k(self, service):
        """Test that rerank respects top_k parameter."""
        results = [Mock(), Mock(), Mock()]
        query = "test query"
        
        reranked_results = await service.rerank(query, results, top_k=2)
        
        assert len(reranked_results) == 2
        assert reranked_results == results[:2]
    
    @pytest.mark.asyncio
    async def test_compute_relevance_scores(self, service):
        """Test that compute_relevance_scores returns zero scores."""
        query = "test query"
        texts = ["text1", "text2", "text3"]
        
        scores = await service.compute_relevance_scores(query, texts)
        
        assert scores == [0.0, 0.0, 0.0]
    
    def test_is_available(self, service):
        """Test that service reports as available."""
        assert service.is_available()
    
    def test_get_model_info(self, service):
        """Test model info for no-op service."""
        info = service.get_model_info()
        
        assert info["model_name"] == "no-op"
        assert info["type"] == "no-operation"


@pytest.mark.integration
class TestRerankingIntegration:
    """Integration tests for reranking within the full search pipeline."""
    
    @pytest.mark.asyncio
    async def test_search_service_with_reranking(self):
        """Test search service integration with reranking."""
        # This would require a full integration test with real models
        # For now, we'll use mocks to test the integration points
        
        from core.domain.search_service import SearchService
        from core.domain.models import SearchQuery, SearchMode
        
        # Mock repositories and services
        vector_repo = Mock()
        reranking_service = Mock()
        
        # Configure mocks
        sample_results = [
            SearchResult(
                text="Test result 1",
                metadata=DocumentMetadata(doc_id="1"),
                score=0.8,
                source_type="vector"
            ),
            SearchResult(
                text="Test result 2", 
                metadata=DocumentMetadata(doc_id="2"),
                score=0.7,
                source_type="vector"
            )
        ]
        
        vector_repo.search.return_value = sample_results
        reranking_service.is_available.return_value = True
        reranking_service.rerank.return_value = list(reversed(sample_results))  # Reversed order
        
        # Create search service
        search_service = SearchService(
            vector_repository=vector_repo,
            reranking_service=reranking_service
        )
        
        # Execute search with reranking
        query = SearchQuery(
            text="test query",
            top_k=2,
            search_mode=SearchMode.VECTOR,
            use_rerank=True
        )
        
        response = await search_service.search(query)
        
        # Verify reranking was called
        reranking_service.rerank.assert_called_once()
        
        # Verify results are from reranking (reversed order)
        assert len(response.results) == 2
        assert response.results[0].metadata.doc_id == "2"
        assert response.results[1].metadata.doc_id == "1"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])