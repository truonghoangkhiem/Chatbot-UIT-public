"""
Cross-encoder reranking service implementation.

This module provides a concrete implementation of the RerankingService port
using sentence-transformers cross-encoder models for improved retrieval accuracy.
"""

import logging
import time
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

from core.ports.services import RerankingService
from core.domain.models import SearchResult, RerankingMetadata

logger = logging.getLogger(__name__)


class CrossEncoderRerankingService(RerankingService):
    """
    Cross-encoder based reranking service using sentence-transformers.
    
    This service uses cross-encoder models to compute query-document relevance scores
    and rerank search results for improved accuracy. Cross-encoders process the query
    and document together, providing more accurate relevance scores than bi-encoders.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        batch_size: int = 16,
        device: Optional[str] = None,
        trust_remote_code: bool = False
    ):
        """
        Initialize the cross-encoder reranking service.
        
        Args:
            model_name: Name of the cross-encoder model to use
            max_length: Maximum input length for the model
            batch_size: Batch size for processing multiple query-document pairs
            device: Device to run the model on ('cpu', 'cuda', etc.)
            trust_remote_code: Whether to trust remote code when loading models
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.trust_remote_code = trust_remote_code
        
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._model_info = {
            "model_name": model_name,
            "max_length": max_length,
            "batch_size": batch_size,
            "device": device or "auto"
        }
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
                trust_remote_code=self.trust_remote_code
            )
            
            logger.info(f"Cross-encoder model loaded successfully on device: {self._model.device}")
            self._model_info["actual_device"] = str(self._model.device)
            
        except ImportError as e:
            logger.error("sentence-transformers library not found. Please install it: pip install sentence-transformers")
            self._model = None
            raise ImportError("sentence-transformers library is required for cross-encoder reranking") from e
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model {self.model_name}: {e}")
            self._model = None
            raise
    
    def is_available(self) -> bool:
        """Check if the reranking service is available."""
        return self._model is not None
    
    def get_model_info(self) -> dict:
        """Get information about the reranking model."""
        return self._model_info.copy()
    
    async def rerank(
        self, 
        query: str, 
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder model.
        
        Args:
            query: The search query
            results: List of search results to rerank
            top_k: Number of top results to return after reranking
            
        Returns:
            List of reranked search results with updated scores and metadata
        """
        if not self.is_available():
            logger.warning("Cross-encoder model not available, returning original results")
            return results
        
        if not results:
            return results
        
        start_time = time.time()
        
        try:
            # Extract texts and prepare query-document pairs
            texts = [result.text for result in results]
            query_doc_pairs = [(query, text) for text in texts]
            
            # Compute relevance scores
            relevance_scores = await self._compute_scores_async(query_doc_pairs)
            
            # Create reranked results with metadata
            reranked_results = []
            for i, (result, score) in enumerate(zip(results, relevance_scores)):
                # Create reranking metadata
                reranking_metadata = RerankingMetadata(
                    original_rank=i + 1,
                    original_score=result.score,
                    rerank_score=float(score),
                    model_name=self.model_name,
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )
                
                # Update result with reranking information
                reranked_result = SearchResult(
                    text=result.text,
                    metadata=result.metadata,
                    score=float(score),  # Use rerank score as primary score
                    source_type=result.source_type,
                    rank=result.rank,
                    char_spans=result.char_spans,
                    highlighted_text=result.highlighted_text,
                    highlighted_title=result.highlighted_title,
                    bm25_score=result.bm25_score,
                    vector_score=result.vector_score,
                    rerank_score=float(score),
                    reranking_metadata=reranking_metadata
                )
                
                reranked_results.append(reranked_result)
            
            # Sort by rerank score (descending)
            reranked_results.sort(key=lambda x: x.rerank_score or 0, reverse=True)
            
            # Update ranks
            for i, result in enumerate(reranked_results):
                result.rank = i + 1
            
            # Apply top_k limit if specified
            if top_k is not None:
                reranked_results = reranked_results[:top_k]
            
            processing_time = int((time.time() - start_time) * 1000)
            logger.info(f"Reranked {len(results)} results in {processing_time}ms")
            
            # Log reranking impact
            if len(results) >= 3:
                original_top3 = [r.metadata.doc_id for r in results[:3]]
                reranked_top3 = [r.metadata.doc_id for r in reranked_results[:3]]
                if original_top3 != reranked_top3:
                    logger.info(f"Reranking changed top-3 results: {original_top3} -> {reranked_top3}")
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Return original results on error
            return results
    
    async def compute_relevance_scores(
        self, 
        query: str, 
        texts: List[str]
    ) -> List[float]:
        """
        Compute relevance scores between query and texts.
        
        Args:
            query: The search query
            texts: List of texts to score
            
        Returns:
            List of relevance scores
        """
        if not self.is_available():
            logger.warning("Cross-encoder model not available, returning zero scores")
            return [0.0] * len(texts)
        
        if not texts:
            return []
        
        query_doc_pairs = [(query, text) for text in texts]
        return await self._compute_scores_async(query_doc_pairs)
    
    async def _compute_scores_async(self, query_doc_pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Compute relevance scores asynchronously.
        
        Args:
            query_doc_pairs: List of (query, document) pairs
            
        Returns:
            List of relevance scores
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._compute_scores_sync,
            query_doc_pairs
        )
    
    def _compute_scores_sync(self, query_doc_pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Compute relevance scores synchronously (runs in thread executor).
        
        Args:
            query_doc_pairs: List of (query, document) pairs
            
        Returns:
            List of relevance scores
        """
        try:
            # Process in batches to manage memory usage
            all_scores = []
            
            for i in range(0, len(query_doc_pairs), self.batch_size):
                batch = query_doc_pairs[i:i + self.batch_size]
                
                # Compute scores for batch
                batch_scores = self._model.predict(batch)
                
                # Convert to list of floats
                if hasattr(batch_scores, 'tolist'):
                    batch_scores = batch_scores.tolist()
                elif not isinstance(batch_scores, list):
                    batch_scores = [float(score) for score in batch_scores]
                
                all_scores.extend(batch_scores)
            
            return all_scores
            
        except Exception as e:
            logger.error(f"Error computing relevance scores: {e}")
            # Return zero scores on error
            return [0.0] * len(query_doc_pairs)
    
    def __del__(self):
        """Cleanup resources when the service is destroyed."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


class MultilingualCrossEncoderRerankingService(CrossEncoderRerankingService):
    """
    Multilingual cross-encoder reranking service optimized for Vietnamese and English.
    
    This service uses multilingual cross-encoder models that can handle Vietnamese
    text effectively, providing better reranking for the UIT chatbot domain.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        vietnamese_model_name: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
        device: Optional[str] = None,
        auto_detect_language: bool = True
    ):
        """
        Initialize the multilingual cross-encoder reranking service.
        
        Args:
            model_name: Default cross-encoder model name
            vietnamese_model_name: Optional Vietnamese-specific model
            max_length: Maximum input length
            batch_size: Batch size for processing
            device: Device to run models on
            auto_detect_language: Whether to auto-detect and route by language
        """
        self.vietnamese_model_name = vietnamese_model_name
        self.auto_detect_language = auto_detect_language
        self._vietnamese_model = None
        
        # Initialize base model
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size,
            device=device
        )
        
        # Initialize Vietnamese-specific model if provided
        if vietnamese_model_name:
            self._initialize_vietnamese_model()
    
    def _initialize_vietnamese_model(self):
        """Initialize Vietnamese-specific cross-encoder model."""
        if not self.vietnamese_model_name:
            return
        
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Loading Vietnamese cross-encoder model: {self.vietnamese_model_name}")
            self._vietnamese_model = CrossEncoder(
                self.vietnamese_model_name,
                max_length=self.max_length,
                device=self.device,
                trust_remote_code=self.trust_remote_code
            )
            
            logger.info("Vietnamese cross-encoder model loaded successfully")
            self._model_info["vietnamese_model"] = self.vietnamese_model_name
            
        except Exception as e:
            logger.error(f"Failed to load Vietnamese cross-encoder model: {e}")
            self._vietnamese_model = None
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection for Vietnamese vs other languages.
        
        Args:
            text: Text to analyze
            
        Returns:
            'vi' for Vietnamese, 'en' for others
        """
        if not self.auto_detect_language:
            return 'en'
        
        # Simple heuristic: if text contains Vietnamese characters, it's Vietnamese
        vietnamese_chars = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ')
        text_chars = set(text.lower())
        
        if vietnamese_chars.intersection(text_chars):
            return 'vi'
        return 'en'
    
    def _select_model(self, query: str, texts: List[str]):
        """
        Select appropriate model based on language detection.
        
        Args:
            query: Search query
            texts: List of document texts
            
        Returns:
            Selected model instance
        """
        if not self._vietnamese_model:
            return self._model
        
        # Detect language from query and first few texts
        sample_texts = [query] + texts[:3]
        vietnamese_count = sum(1 for text in sample_texts if self._detect_language(text) == 'vi')
        
        # Use Vietnamese model if majority of samples are Vietnamese
        if vietnamese_count > len(sample_texts) / 2:
            logger.debug("Using Vietnamese cross-encoder model")
            return self._vietnamese_model
        else:
            logger.debug("Using default cross-encoder model")
            return self._model
    
    def _compute_scores_sync(self, query_doc_pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Compute relevance scores with model selection based on language.
        
        Args:
            query_doc_pairs: List of (query, document) pairs
            
        Returns:
            List of relevance scores
        """
        if not query_doc_pairs:
            return []
        
        try:
            # Select appropriate model
            query = query_doc_pairs[0][0]  # All pairs have same query
            texts = [pair[1] for pair in query_doc_pairs]
            selected_model = self._select_model(query, texts)
            
            # Process in batches
            all_scores = []
            
            for i in range(0, len(query_doc_pairs), self.batch_size):
                batch = query_doc_pairs[i:i + self.batch_size]
                
                # Compute scores for batch
                batch_scores = selected_model.predict(batch)
                
                # Convert to list of floats
                if hasattr(batch_scores, 'tolist'):
                    batch_scores = batch_scores.tolist()
                elif not isinstance(batch_scores, list):
                    batch_scores = [float(score) for score in batch_scores]
                
                all_scores.extend(batch_scores)
            
            return all_scores
            
        except Exception as e:
            logger.error(f"Error computing relevance scores with multilingual model: {e}")
            return [0.0] * len(query_doc_pairs)


def create_reranking_service(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    multilingual: bool = True,
    **kwargs
) -> RerankingService:
    """
    Factory function to create appropriate reranking service.
    
    Args:
        model_name: Cross-encoder model name
        multilingual: Whether to use multilingual service
        **kwargs: Additional arguments for service initialization
        
    Returns:
        RerankingService instance
    """
    try:
        if multilingual:
            return MultilingualCrossEncoderRerankingService(
                model_name=model_name,
                **kwargs
            )
        else:
            return CrossEncoderRerankingService(
                model_name=model_name,
                **kwargs
            )
    except Exception as e:
        logger.error(f"Failed to create reranking service: {e}")
        # Return a no-op service that doesn't rerank
        return NoOpRerankingService()


class NoOpRerankingService(RerankingService):
    """No-operation reranking service that returns results unchanged."""
    
    async def rerank(
        self, 
        query: str, 
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """Return results unchanged."""
        return results[:top_k] if top_k else results
    
    async def compute_relevance_scores(
        self, 
        query: str, 
        texts: List[str]
    ) -> List[float]:
        """Return zero scores."""
        return [0.0] * len(texts)
    
    def is_available(self) -> bool:
        """Always available but doesn't actually rerank."""
        return True
    
    def get_model_info(self) -> dict:
        """Return no-op model info."""
        return {"model_name": "no-op", "type": "no-operation"}