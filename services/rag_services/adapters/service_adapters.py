# adapters/service_adapters.py
#
# Description:
# Adapter implementations for various external services like reranking, fusion, etc.
# These adapters implement the service ports defined in the core domain.

import logging
from typing import List
from sentence_transformers import CrossEncoder

from core.ports.services import RerankingService, FusionService
from core.domain.models import SearchResult
from core.domain.fusion_service import FusionAlgorithms

logger = logging.getLogger(__name__)


class CrossEncoderRerankingAdapter(RerankingService):
    """
    Adapter for reranking using CrossEncoder models.
    
    This adapter wraps the CrossEncoder functionality to implement
    the RerankingService port from the core domain.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the CrossEncoder model."""
        try:
            if CrossEncoder is not None:
                self._model = CrossEncoder(self.model_name)
                logger.info(f"Initialized CrossEncoder model: {self.model_name}")
            else:
                logger.warning("CrossEncoder not available")
        except Exception as e:
            logger.error(f"Failed to initialize CrossEncoder model: {e}")
            self._model = None
    
    async def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank search results based on query-result relevance."""
        if not self.is_available() or not results:
            return results
        
        try:
            # Prepare query-text pairs for the cross-encoder
            pairs = [(query, result.text) for result in results]
            
            # Get rerank scores
            rerank_scores = self._model.predict(pairs)
            
            # Update results with rerank scores and sort
            reranked_results = []
            for result, score in zip(results, rerank_scores):
                # Create a new result with updated score
                reranked_result = SearchResult(
                    text=result.text,
                    metadata=result.metadata,
                    score=float(score),
                    source_type=f"{result.source_type}_reranked",
                    rank=result.rank,
                    char_spans=result.char_spans,
                    highlighted_text=result.highlighted_text,
                    highlighted_title=result.highlighted_title,
                    bm25_score=result.bm25_score,
                    vector_score=result.vector_score,
                    rerank_score=float(score)
                )
                reranked_results.append(reranked_result)
            
            # Sort by rerank score
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"Reranked {len(results)} results using CrossEncoder")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            return results
    
    def is_available(self) -> bool:
        """Check if the reranking service is available."""
        return self._model is not None


class HybridFusionAdapter(FusionService):
    """
    Adapter for search result fusion using RRF (Reciprocal Rank Fusion).
    
    This adapter uses pure domain logic from FusionAlgorithms to implement
    the FusionService port, following clean architecture principles.
    """
    
    def __init__(self, rrf_constant: int = 60):
        """
        Initialize the fusion adapter.
        
        Args:
            rrf_constant: RRF constant k (typically 60)
        """
        self.rrf_constant = rrf_constant
    
    async def fuse_results(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        rrf_constant: int = 60
    ) -> List[SearchResult]:
        """
        Fuse results from different search methods using RRF.
        
        This adapter delegates to pure domain logic in FusionAlgorithms,
        maintaining clean separation between domain and infrastructure.
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword/BM25 search
            vector_weight: Weight for vector results (default 0.5)
            keyword_weight: Weight for keyword results (default 0.5)
            rrf_constant: RRF constant k (default from initialization)
            
        Returns:
            List of fused search results
        """
        try:
            # Use pure domain logic - no conversion needed!
            fused_results = FusionAlgorithms.reciprocal_rank_fusion(
                vector_results=vector_results,
                keyword_results=keyword_results,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
                rrf_constant=rrf_constant or self.rrf_constant
            )
            
            logger.info(
                f"Fused {len(vector_results)} vector + {len(keyword_results)} keyword "
                f"results into {len(fused_results)} results using domain RRF algorithm"
            )
            
            return fused_results
            
        except Exception as e:
            logger.error(f"Error in result fusion: {e}")
            # Return combined results as fallback
            combined = vector_results + keyword_results
            combined.sort(key=lambda x: x.score, reverse=True)
            return combined[:max(len(vector_results), len(keyword_results))]
