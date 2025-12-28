"""
Graph Builder Configuration.

This module defines configuration for the GraphBuilderService.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class DeduplicationStrategy(str, Enum):
    """Strategy for entity deduplication"""
    EXACT_MATCH = "exact_match"  # Exact string match (fast)
    FUZZY_MATCH = "fuzzy_match"  # Fuzzy string matching
    EMBEDDING = "embedding"  # Embedding-based similarity
    HYBRID = "hybrid"  # Combination of methods


class ConflictResolutionStrategy(str, Enum):
    """Strategy for resolving conflicting entities"""
    MERGE = "merge"  # Merge all properties
    KEEP_FIRST = "keep_first"  # Keep first occurrence
    KEEP_HIGHEST_CONFIDENCE = "keep_highest_confidence"  # Keep entity with highest confidence
    MANUAL_REVIEW = "manual_review"  # Flag for manual review


@dataclass
class BatchConfig:
    """Configuration for batch operations"""
    batch_size: int = 100
    max_concurrent_batches: int = 5
    retry_failed_batches: bool = True
    max_retries: int = 3
    backoff_factor: float = 2.0  # Exponential backoff


@dataclass
class DeduplicationConfig:
    """Configuration for entity deduplication"""
    strategy: DeduplicationStrategy = DeduplicationStrategy.FUZZY_MATCH
    fuzzy_threshold: float = 0.9  # Similarity threshold for fuzzy matching (0-1)
    embedding_threshold: float = 0.85  # Similarity threshold for embeddings
    normalize_text: bool = True  # Normalize Vietnamese text before comparison
    case_sensitive: bool = False


@dataclass
class ValidationConfig:
    """Configuration for entity and relationship validation"""
    validate_schema: bool = True  # Validate against CatRAG schema
    validate_duplicates: bool = True  # Check for duplicates before insertion
    validate_references: bool = True  # Validate that referenced nodes exist
    min_confidence_threshold: float = 0.7  # Minimum confidence for entities/relations
    strict_mode: bool = False  # Raise errors vs. warnings


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    enable_caching: bool = True
    cache_size: int = 1000
    enable_parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_mb: int = 500  # Memory limit for graph building


@dataclass
class GraphBuilderConfig:
    """
    Comprehensive configuration for GraphBuilderService.
    
    This configuration controls all aspects of graph building:
    - Batch processing parameters
    - Deduplication strategies
    - Validation rules
    - Performance optimizations
    
    Example:
        ```python
        config = GraphBuilderConfig(
            batch_size=200,
            deduplication_strategy=DeduplicationStrategy.HYBRID,
            min_confidence_threshold=0.8
        )
        
        service = GraphBuilderService(
            graph_repo=neo4j_repo,
            entity_extractor=extractor,
            config=config
        )
        ```
    """
    
    # Batch processing
    batch_size: int = 100
    batch_config: BatchConfig = field(default_factory=BatchConfig)
    
    # Deduplication
    deduplication_strategy: DeduplicationStrategy = DeduplicationStrategy.FUZZY_MATCH
    deduplication_config: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    
    # Conflict resolution
    conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.KEEP_HIGHEST_CONFIDENCE
    
    # Validation
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Performance
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Entity extraction
    extract_entities_from_documents: bool = True
    extract_relations_from_documents: bool = True
    use_llm_for_relations: bool = True  # Use LLM for relation extraction (Week 2)
    
    # Logging and monitoring
    verbose: bool = True
    log_progress_interval: int = 100  # Log progress every N items
    collect_metrics: bool = True
    
    # Graph-specific
    create_indexes: bool = True  # Create Neo4j indexes automatically
    enable_constraints: bool = True  # Enable uniqueness constraints
    
    def __post_init__(self):
        """Validate configuration"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if not 0 <= self.deduplication_config.fuzzy_threshold <= 1:
            raise ValueError("fuzzy_threshold must be between 0 and 1")
        
        if not 0 <= self.validation_config.min_confidence_threshold <= 1:
            raise ValueError("min_confidence_threshold must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "batch_size": self.batch_size,
            "deduplication_strategy": self.deduplication_strategy.value,
            "conflict_resolution": self.conflict_resolution.value,
            "min_confidence_threshold": self.validation_config.min_confidence_threshold,
            "use_llm_for_relations": self.use_llm_for_relations,
            "verbose": self.verbose,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphBuilderConfig':
        """Create config from dictionary"""
        return cls(
            batch_size=data.get("batch_size", 100),
            deduplication_strategy=DeduplicationStrategy(
                data.get("deduplication_strategy", "fuzzy_match")
            ),
            conflict_resolution=ConflictResolutionStrategy(
                data.get("conflict_resolution", "keep_highest_confidence")
            ),
            use_llm_for_relations=data.get("use_llm_for_relations", True),
            verbose=data.get("verbose", True),
        )
    
    @classmethod
    def default(cls) -> 'GraphBuilderConfig':
        """Get default configuration"""
        return cls()
    
    @classmethod
    def high_performance(cls) -> 'GraphBuilderConfig':
        """Configuration optimized for high performance"""
        return cls(
            batch_size=500,
            batch_config=BatchConfig(
                batch_size=500,
                max_concurrent_batches=10,
            ),
            deduplication_strategy=DeduplicationStrategy.EXACT_MATCH,  # Faster
            performance_config=PerformanceConfig(
                enable_parallel_processing=True,
                max_workers=8,
                memory_limit_mb=2000,
            ),
            validation_config=ValidationConfig(
                validate_duplicates=False,  # Skip for speed
                strict_mode=False,
            ),
        )
    
    @classmethod
    def high_quality(cls) -> 'GraphBuilderConfig':
        """Configuration optimized for data quality"""
        return cls(
            batch_size=50,  # Smaller batches for better error handling
            deduplication_strategy=DeduplicationStrategy.HYBRID,
            conflict_resolution=ConflictResolutionStrategy.MANUAL_REVIEW,
            deduplication_config=DeduplicationConfig(
                strategy=DeduplicationStrategy.HYBRID,
                fuzzy_threshold=0.95,  # More strict
                normalize_text=True,
            ),
            validation_config=ValidationConfig(
                validate_schema=True,
                validate_duplicates=True,
                validate_references=True,
                min_confidence_threshold=0.8,  # Higher threshold
                strict_mode=True,
            ),
            use_llm_for_relations=True,
            verbose=True,
        )
