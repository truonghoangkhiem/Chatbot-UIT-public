"""
IRCoT (Interleaving Retrieval with Chain-of-Thought) Configuration.

This module provides configuration settings for the IRCoT reasoning approach,
which enables dynamic retrieval during chain-of-thought reasoning for complex queries.

Reference: "Interleaving Retrieval with Chain-of-Thought Reasoning for 
            Knowledge-Intensive Multi-Step Questions" (Trivedi et al., 2022)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class IRCoTMode(Enum):
    """IRCoT execution modes."""
    DISABLED = "disabled"           # Traditional one-shot retrieval
    AUTOMATIC = "automatic"         # Auto-enable based on complexity
    FORCED = "forced"               # Always use IRCoT


@dataclass
class IRCoTConfig:
    """Configuration for IRCoT (Interleaving Retrieval with Chain-of-Thought) reasoning.
    
    IRCoT improves answer quality for complex, multi-hop questions by:
    1. Breaking down complex queries into reasoning steps
    2. Dynamically retrieving new context based on each reasoning step
    3. Iteratively building a comprehensive understanding
    
    Attributes:
        enabled: Whether IRCoT is enabled
        mode: IRCoT execution mode (disabled, automatic, forced)
        max_iterations: Maximum number of reasoning-retrieval iterations
        min_iterations: Minimum iterations before early stopping
        complexity_threshold: Minimum complexity score to trigger IRCoT in automatic mode
        context_accumulation: Whether to accumulate context across iterations
        max_context_documents: Maximum total documents to accumulate
        retrieval_top_k: Number of documents to retrieve per iteration
        cot_temperature: Temperature for Chain-of-Thought generation
        early_stopping_threshold: Confidence threshold for early stopping
        deduplication_enabled: Whether to deduplicate retrieved documents
        reasoning_prompt_template: Template for generating CoT reasoning prompts
    """
    
    # Core settings
    enabled: bool = True
    mode: IRCoTMode = IRCoTMode.AUTOMATIC
    
    # Iteration limits - OPTIMIZED: reduced from 3 to 2 for faster response
    # Each iteration costs ~5s (RAG + LLM), so 2 iterations = ~10s vs 3 iterations = ~15s
    max_iterations: int = 2
    min_iterations: int = 1
    
    # Complexity threshold for automatic mode
    complexity_threshold: float = 6.5  # Same as SmartPlannerAgent.COMPLEX_MIN_SCORE
    
    # Context management
    context_accumulation: bool = True
    max_context_documents: int = 15
    retrieval_top_k: int = 3  # Per iteration
    
    # LLM settings
    cot_temperature: float = 0.3
    cot_model: Optional[str] = None  # Model for CoT reasoning (None = use agent's default)
    
    # Early stopping - OPTIMIZED for faster response
    early_stopping_enabled: bool = True
    early_stopping_threshold: float = 0.70  # Reduced from 0.85 for faster early stopping
    
    # Document handling
    deduplication_enabled: bool = True
    document_similarity_threshold: float = 0.9  # For deduplication
    
    # Prompts
    reasoning_prompt_template: str = field(default_factory=lambda: """
Dựa trên thông tin đã có và câu hỏi của người dùng, hãy thực hiện một bước suy luận.

**Câu hỏi gốc:** {original_query}

**Thông tin đã thu thập:**
{accumulated_context}

**Bước suy luận hiện tại:** {current_step}/{max_steps}

Hãy:
1. Phân tích thông tin hiện có
2. Xác định thông tin còn thiếu để trả lời đầy đủ câu hỏi
3. Đưa ra một câu suy luận/kết luận trung gian
4. Nếu cần thêm thông tin, đề xuất câu truy vấn tìm kiếm tiếp theo

Trả lời theo format JSON:
{{
    "reasoning_step": "<Bước suy luận/kết luận trung gian>",
    "information_gaps": ["<Thông tin còn thiếu 1>", "<Thông tin còn thiếu 2>"],
    "next_search_query": "<Câu truy vấn để tìm thêm thông tin, hoặc null nếu đủ>",
    "confidence": <0.0-1.0>,
    "can_answer_now": <true/false>
}}
""")
    
    def should_use_ircot(self, complexity: str, complexity_score: float) -> bool:
        """
        Determine if IRCoT should be used based on complexity.
        
        Args:
            complexity: Complexity level ("simple", "medium", "complex")
            complexity_score: Numeric complexity score (0-10)
            
        Returns:
            True if IRCoT should be activated
        """
        if not self.enabled:
            return False
            
        if self.mode == IRCoTMode.DISABLED:
            return False
            
        if self.mode == IRCoTMode.FORCED:
            return True
            
        # Automatic mode: check complexity
        return (complexity == "complex" or 
                complexity_score >= self.complexity_threshold)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enabled": self.enabled,
            "mode": self.mode.value,
            "max_iterations": self.max_iterations,
            "min_iterations": self.min_iterations,
            "complexity_threshold": self.complexity_threshold,
            "context_accumulation": self.context_accumulation,
            "max_context_documents": self.max_context_documents,
            "retrieval_top_k": self.retrieval_top_k,
            "cot_temperature": self.cot_temperature,
            "early_stopping_enabled": self.early_stopping_enabled,
            "early_stopping_threshold": self.early_stopping_threshold,
            "deduplication_enabled": self.deduplication_enabled
        }


@dataclass
class IRCoTIterationResult:
    """Result of a single IRCoT iteration.
    
    Attributes:
        iteration: The iteration number (1-indexed)
        reasoning_step: The reasoning/conclusion from this iteration
        next_search_query: Query for next retrieval (None if complete)
        retrieved_documents: Documents retrieved in this iteration
        confidence: Confidence level after this iteration
        can_answer_now: Whether enough information has been gathered
        information_gaps: List of identified information gaps
    """
    iteration: int
    reasoning_step: str
    next_search_query: Optional[str]
    retrieved_documents: List[Dict[str, Any]]
    confidence: float
    can_answer_now: bool
    information_gaps: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "iteration": self.iteration,
            "reasoning_step": self.reasoning_step,
            "next_search_query": self.next_search_query,
            "documents_count": len(self.retrieved_documents),
            "confidence": self.confidence,
            "can_answer_now": self.can_answer_now,
            "information_gaps": self.information_gaps
        }


@dataclass 
class IRCoTResult:
    """Final result of IRCoT reasoning process.
    
    Attributes:
        original_query: The original user query
        iterations: List of iteration results
        total_iterations: Number of iterations performed
        accumulated_context: All accumulated context documents
        final_reasoning: Combined reasoning from all iterations
        final_confidence: Final confidence score
        early_stopped: Whether reasoning stopped early
        processing_time: Total processing time in seconds
    """
    original_query: str
    iterations: List[IRCoTIterationResult]
    total_iterations: int
    accumulated_context: List[Dict[str, Any]]
    final_reasoning: str
    final_confidence: float
    early_stopped: bool = False
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "original_query": self.original_query,
            "iterations": [it.to_dict() for it in self.iterations],
            "total_iterations": self.total_iterations,
            "accumulated_documents_count": len(self.accumulated_context),
            "final_reasoning": self.final_reasoning,
            "final_confidence": self.final_confidence,
            "early_stopped": self.early_stopped,
            "processing_time": self.processing_time
        }
    
    def get_all_reasoning_steps(self) -> List[str]:
        """Get all reasoning steps from iterations."""
        return [it.reasoning_step for it in self.iterations if it.reasoning_step]
    
    def get_all_search_queries(self) -> List[str]:
        """Get all search queries used (including original)."""
        queries = [self.original_query]
        for it in self.iterations:
            if it.next_search_query:
                queries.append(it.next_search_query)
        return queries


# Default configuration instance
DEFAULT_IRCOT_CONFIG = IRCoTConfig()
