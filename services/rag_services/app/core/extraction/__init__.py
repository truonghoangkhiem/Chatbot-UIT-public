"""
Extraction module for Knowledge Graph extraction pipeline.

This module provides:
- Schema definitions (models, enums, prompts)
- Structure extraction (VLM-based)
- Semantic extraction (LLM-based)
- Hybrid extraction pipeline
"""

from app.core.extraction.schemas import (
    # Enums
    StructureNodeType,
    VLMProvider,
    # Structural models
    StructureNode,
    StructureRelation,
    StructureExtractionResult,
    # Semantic models
    SemanticNode,
    SemanticRelation,
    SemanticExtractionResult,
    # Combined models
    HybridExtractionResult,
    PageContext,
    # Config models
    VLMConfig,
    LLMConfig,
    # Schema definitions
    VALID_ENTITY_TYPES,
    VALID_RELATION_TYPES,
    UNIFIED_ACADEMIC_SCHEMA,
    STRUCTURE_EXTRACTION_PROMPT,
    SEMANTIC_EXTRACTION_PROMPT,
)

from app.core.extraction.hybrid_extractor import (
    StructureExtractor,
    SemanticExtractor,
    ParallelSemanticExtractor,
    convert_to_graph_models,
    run_pipeline,
)

__all__ = [
    # Enums
    "StructureNodeType",
    "VLMProvider",
    # Structural models
    "StructureNode",
    "StructureRelation",
    "StructureExtractionResult",
    # Semantic models
    "SemanticNode",
    "SemanticRelation",
    "SemanticExtractionResult",
    # Combined models
    "HybridExtractionResult",
    "PageContext",
    # Config models
    "VLMConfig",
    "LLMConfig",
    # Schema definitions
    "VALID_ENTITY_TYPES",
    "VALID_RELATION_TYPES",
    "UNIFIED_ACADEMIC_SCHEMA",
    "STRUCTURE_EXTRACTION_PROMPT",
    "SEMANTIC_EXTRACTION_PROMPT",
    # Extractors
    "StructureExtractor",
    "SemanticExtractor",
    "ParallelSemanticExtractor",
    # Functions
    "convert_to_graph_models",
    "run_pipeline",
]
