"""
Domain service for result fusion logic.

This module contains pure business logic for fusing search results from
different sources (vector search, keyword search). This is domain logic
and should not have any infrastructure dependencies.
"""

from typing import List, Dict, Set
from .models import SearchResult


class FusionAlgorithms:
    """
    Pure domain logic for result fusion.
    
    This class contains algorithms for combining search results from
    different sources without any infrastructure dependencies.
    """
    
    @staticmethod
    def reciprocal_rank_fusion(
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        rrf_constant: int = 60
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF Formula: score = w1/(k + rank1) + w2/(k + rank2)
        
        This is pure business logic without infrastructure dependencies.
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword/BM25 search
            vector_weight: Weight for vector results (default 0.5)
            keyword_weight: Weight for keyword results (default 0.5)
            rrf_constant: RRF constant k (typically 60)
            
        Returns:
            List of fused search results, sorted by relevance
        """
        # Helper function to create unique key for a result
        def create_key(result: SearchResult) -> str:
            """Create unique key for deduplication."""
            return f"{result.metadata.doc_id}_{result.metadata.chunk_id or ''}"
        
        # Create result maps for efficient lookup
        vector_map: Dict[str, tuple[int, SearchResult]] = {
            create_key(r): (i + 1, r) 
            for i, r in enumerate(vector_results)
        }
        
        keyword_map: Dict[str, tuple[int, SearchResult]] = {
            create_key(r): (i + 1, r) 
            for i, r in enumerate(keyword_results)
        }
        
        # Get all unique document keys
        all_keys: Set[str] = set(vector_map.keys()) | set(keyword_map.keys())
        
        fused_results: List[SearchResult] = []
        
        for key in all_keys:
            rrf_score = 0.0
            result = None
            original_vector_score = None
            original_bm25_score = None
            
            # Add contribution from keyword results
            if key in keyword_map:
                rank, kw_result = keyword_map[key]
                rrf_score += keyword_weight / (rrf_constant + rank)
                result = kw_result
                original_bm25_score = kw_result.score
                
            # Add contribution from vector results
            if key in vector_map:
                rank, vec_result = vector_map[key]
                rrf_score += vector_weight / (rrf_constant + rank)
                original_vector_score = vec_result.score
                # If we don't have result from keyword, use vector result
                if result is None:
                    result = vec_result
            
            if result:
                # Create new fused result with combined score
                fused_result = SearchResult(
                    text=result.text,
                    metadata=result.metadata,
                    score=rrf_score,  # Use RRF score as primary score
                    source_type="fused",
                    rank=None,  # Will be set after sorting
                    char_spans=result.char_spans,
                    highlighted_text=result.highlighted_text,
                    highlighted_title=result.highlighted_title,
                    bm25_score=original_bm25_score,
                    vector_score=original_vector_score,
                    rerank_score=result.rerank_score
                )
                fused_results.append(fused_result)
        
        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        # Add rank information
        for i, result in enumerate(fused_results):
            result.rank = i + 1
        
        return fused_results
    
    @staticmethod
    def weighted_score_fusion(
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5
    ) -> List[SearchResult]:
        """
        Combine results using weighted score fusion.
        
        Formula: score = w1*score1 + w2*score2
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword/BM25 search
            vector_weight: Weight for vector scores
            keyword_weight: Weight for keyword scores
            
        Returns:
            List of fused search results, sorted by relevance
        """
        def create_key(result: SearchResult) -> str:
            return f"{result.metadata.doc_id}_{result.metadata.chunk_id or ''}"
        
        # Normalize scores to [0, 1] range
        def normalize_scores(results: List[SearchResult]) -> Dict[str, float]:
            if not results:
                return {}
            
            max_score = max(r.score for r in results)
            min_score = min(r.score for r in results)
            
            # Avoid division by zero
            if max_score == min_score:
                return {create_key(r): 1.0 for r in results}
            
            return {
                create_key(r): (r.score - min_score) / (max_score - min_score)
                for r in results
            }
        
        # Normalize scores
        vector_scores = normalize_scores(vector_results)
        keyword_scores = normalize_scores(keyword_results)
        
        # Create result maps
        vector_map = {create_key(r): r for r in vector_results}
        keyword_map = {create_key(r): r for r in keyword_results}
        
        all_keys = set(vector_map.keys()) | set(keyword_map.keys())
        
        fused_results = []
        
        for key in all_keys:
            weighted_score = 0.0
            result = None
            
            if key in vector_scores:
                weighted_score += vector_weight * vector_scores[key]
                result = vector_map[key]
            
            if key in keyword_scores:
                weighted_score += keyword_weight * keyword_scores[key]
                if result is None:
                    result = keyword_map[key]
            
            if result:
                fused_result = SearchResult(
                    text=result.text,
                    metadata=result.metadata,
                    score=weighted_score,
                    source_type="fused",
                    rank=None,
                    char_spans=result.char_spans,
                    highlighted_text=result.highlighted_text,
                    highlighted_title=result.highlighted_title,
                    bm25_score=result.bm25_score,
                    vector_score=result.vector_score
                )
                fused_results.append(fused_result)
        
        # Sort and rank
        fused_results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(fused_results):
            result.rank = i + 1
        
        return fused_results
