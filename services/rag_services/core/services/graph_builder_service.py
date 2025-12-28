"""
Graph Builder Service - Core Implementation.

This service handles building and populating the knowledge graph from documents.
It orchestrates entity extraction, deduplication, relationship creation, and
batch operations for efficient graph population.

Week 2 - Task A1: Graph Builder Core Service
Priority: P0 (Critical)

Architecture:
    GraphBuilderService
    ├── EntityProcessor (validates & transforms entities)
    ├── RelationshipProcessor (creates relationships)
    ├── BatchProcessor (handles large-scale operations)
    └── ConflictResolver (handles duplicates)
"""

import logging
import asyncio
from collections import defaultdict
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime

from core.domain.graph_models import (
    GraphNode,
    GraphRelationship,
    NodeCategory,
    RelationshipType,
    Entity,
    Relation,
)
from core.ports.graph_repository import GraphRepository
# NOTE: Old indexing module removed - entity extraction now handled differently
# from indexing.category_guided_entity_extractor import CategoryGuidedEntityExtractor
from .graph_builder_config import (
    GraphBuilderConfig,
    DeduplicationStrategy,
    ConflictResolutionStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphBuildResult:
    """Result of graph building operation"""
    created_nodes: int = 0
    created_relationships: int = 0
    deduplicated_entities: int = 0
    failed_nodes: int = 0
    failed_relationships: int = 0
    processing_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def success_rate(self) -> float:
        """Calculate success rate for nodes"""
        total = self.created_nodes + self.failed_nodes
        if total == 0:
            return 1.0
        return self.created_nodes / total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "created_nodes": self.created_nodes,
            "created_relationships": self.created_relationships,
            "deduplicated_entities": self.deduplicated_entities,
            "failed_nodes": self.failed_nodes,
            "failed_relationships": self.failed_relationships,
            "processing_time_seconds": self.processing_time_seconds,
            "success_rate": self.success_rate(),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


@dataclass
class Document:
    """Simple document representation"""
    content: str
    doc_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class EntityProcessor:
    """Validates and transforms entities"""
    
    def __init__(self, config: GraphBuilderConfig):
        self.config = config
    
    def validate(self, entity: Entity) -> Tuple[bool, Optional[str]]:
        """
        Validate entity against rules.
        
        Returns:
            (is_valid, error_message)
        """
        # Check confidence threshold
        if entity.confidence < self.config.validation_config.min_confidence_threshold:
            return False, f"Confidence {entity.confidence} below threshold"
        
        # Check required fields
        if not entity.text or not entity.text.strip():
            return False, "Entity text is empty"
        
        # Check category is valid
        try:
            NodeCategory(entity.type)
        except ValueError:
            return False, f"Invalid category: {entity.type}"
        
        return True, None
    
    def normalize(self, entity: Entity) -> Entity:
        """
        Normalize entity text and properties.
        
        For Vietnamese text:
        - Remove extra whitespace
        - Standardize diacritics
        - Convert to standard form
        """
        if self.config.deduplication_config.normalize_text:
            # Normalize whitespace
            normalized_text = " ".join(entity.text.split())
            
            # Strip punctuation from ends
            normalized_text = normalized_text.strip(".,;:!?")
            
            # Case normalization
            if not self.config.deduplication_config.case_sensitive:
                normalized_text = normalized_text.lower()
            
            entity.normalized = normalized_text
        else:
            entity.normalized = entity.text
        
        return entity
    
    def entity_to_node(self, entity: Entity) -> GraphNode:
        """
        Convert Entity to GraphNode.
        
        Maps extracted entities to graph nodes with proper properties.
        """
        category = NodeCategory(entity.type)
        
        # Base properties
        properties = {
            "name": entity.text,
            "normalized_name": entity.normalized or entity.text,
            "confidence": entity.confidence,
            "extraction_metadata": entity.metadata,
        }
        
        # Category-specific properties
        if category == NodeCategory.MON_HOC:
            # Try to extract course code from metadata
            code = entity.metadata.get("code", entity.text)
            properties["code"] = code
            properties["credits"] = entity.metadata.get("credits", 0)
        
        elif category == NodeCategory.QUY_DINH:
            properties["id"] = entity.metadata.get("id", entity.text)
            properties["title"] = entity.text
            properties["year"] = entity.metadata.get("year", datetime.now().year)
        
        elif category == NodeCategory.KHOA:
            properties["code"] = entity.metadata.get("code", entity.text)
        
        return GraphNode(
            category=category,
            properties=properties
        )


class RelationshipProcessor:
    """Creates and validates relationships"""
    
    def __init__(self, config: GraphBuilderConfig):
        self.config = config
    
    def validate(self, relation: Relation) -> Tuple[bool, Optional[str]]:
        """
        Validate relationship.
        
        Returns:
            (is_valid, error_message)
        """
        # Check confidence
        if relation.confidence < self.config.validation_config.min_confidence_threshold:
            return False, f"Confidence {relation.confidence} below threshold"
        
        # Check relationship type is valid
        try:
            RelationshipType(relation.rel_type)
        except ValueError:
            return False, f"Invalid relationship type: {relation.rel_type}"
        
        # Check entities are valid
        if not relation.source or not relation.target:
            return False, "Missing source or target entity"
        
        # Check for self-referencing (if same entity)
        if (relation.source.text == relation.target.text and 
            relation.source.type == relation.target.type):
            return False, "Self-referencing relationship"
        
        return True, None
    
    def relation_to_graph_relationship(
        self,
        relation: Relation,
        entity_to_node_id: Dict[str, str]
    ) -> Optional[GraphRelationship]:
        """
        Convert Relation to GraphRelationship.
        
        Args:
            relation: Extracted relation
            entity_to_node_id: Mapping from entity key to node ID
            
        Returns:
            GraphRelationship or None if mapping fails
        """
        # Get node IDs for source and target
        source_key = self._entity_key(relation.source)
        target_key = self._entity_key(relation.target)
        
        source_id = entity_to_node_id.get(source_key)
        target_id = entity_to_node_id.get(target_key)
        
        if not source_id or not target_id:
            logger.warning(f"Cannot map relation to nodes: {source_key} -> {target_key}")
            return None
        
        return GraphRelationship(
            source_id=source_id,
            target_id=target_id,
            rel_type=RelationshipType(relation.rel_type),
            properties={
                "confidence": relation.confidence,
                **relation.metadata
            }
        )
    
    def _entity_key(self, entity: Entity) -> str:
        """Create unique key for entity"""
        normalized = entity.normalized or entity.text
        return f"{entity.type}:{normalized}"


class ConflictResolver:
    """Handles duplicate entities and conflicts"""
    
    def __init__(self, config: GraphBuilderConfig):
        self.config = config
    
    def resolve_duplicates(
        self,
        entities: List[Entity]
    ) -> List[Entity]:
        """
        Resolve duplicate entities based on strategy.
        
        Args:
            entities: List of potentially duplicate entities
            
        Returns:
            Deduplicated list of entities
        """
        strategy = self.config.deduplication_config.strategy
        
        if strategy == DeduplicationStrategy.EXACT_MATCH:
            return self._exact_match_dedup(entities)
        elif strategy == DeduplicationStrategy.FUZZY_MATCH:
            return self._fuzzy_match_dedup(entities)
        elif strategy == DeduplicationStrategy.EMBEDDING:
            return self._embedding_dedup(entities)
        elif strategy == DeduplicationStrategy.HYBRID:
            return self._hybrid_dedup(entities)
        else:
            logger.warning(f"Unknown dedup strategy: {strategy}, using exact match")
            return self._exact_match_dedup(entities)
    
    def _exact_match_dedup(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate using exact string matching"""
        seen: Dict[str, Entity] = {}
        
        for entity in entities:
            key = f"{entity.type}:{entity.normalized or entity.text}"
            
            if key not in seen:
                seen[key] = entity
            else:
                # Merge with existing
                existing = seen[key]
                seen[key] = self._merge_entities(existing, entity)
        
        return list(seen.values())
    
    def _fuzzy_match_dedup(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate using fuzzy string matching"""
        try:
            from fuzzywuzzy import fuzz
        except ImportError:
            logger.warning("fuzzywuzzy not installed, falling back to exact match")
            return self._exact_match_dedup(entities)
        
        # Group by category first
        by_category: Dict[str, List[Entity]] = defaultdict(list)
        for entity in entities:
            by_category[entity.type].append(entity)
        
        # Deduplicate within each category
        result = []
        threshold = self.config.deduplication_config.fuzzy_threshold * 100
        
        for category, cat_entities in by_category.items():
            deduplicated = []
            
            for entity in cat_entities:
                # Find best match in deduplicated list
                best_match = None
                best_score = 0
                
                for existing in deduplicated:
                    score = fuzz.ratio(
                        entity.normalized or entity.text,
                        existing.normalized or existing.text
                    )
                    if score > best_score:
                        best_score = score
                        best_match = existing
                
                if best_match and best_score >= threshold:
                    # Merge with best match
                    idx = deduplicated.index(best_match)
                    deduplicated[idx] = self._merge_entities(best_match, entity)
                else:
                    # Add as new entity
                    deduplicated.append(entity)
            
            result.extend(deduplicated)
        
        return result
    
    def _embedding_dedup(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate using embedding similarity (stub for Week 2)"""
        logger.warning("Embedding dedup not yet implemented, using fuzzy match")
        return self._fuzzy_match_dedup(entities)
    
    def _hybrid_dedup(self, entities: List[Entity]) -> List[Entity]:
        """Hybrid approach: exact + fuzzy"""
        # First pass: exact match
        exact_deduped = self._exact_match_dedup(entities)
        
        # Second pass: fuzzy match on results
        return self._fuzzy_match_dedup(exact_deduped)
    
    def _merge_entities(self, e1: Entity, e2: Entity) -> Entity:
        """
        Merge two entities based on conflict resolution strategy.
        
        Args:
            e1: First entity
            e2: Second entity
            
        Returns:
            Merged entity
        """
        strategy = self.config.conflict_resolution
        
        if strategy == ConflictResolutionStrategy.KEEP_HIGHEST_CONFIDENCE:
            if e2.confidence > e1.confidence:
                return e2
            return e1
        
        elif strategy == ConflictResolutionStrategy.KEEP_FIRST:
            return e1
        
        elif strategy == ConflictResolutionStrategy.MERGE:
            # Merge properties
            merged_metadata = {**e1.metadata, **e2.metadata}
            merged_metadata["merged_from"] = [e1.text, e2.text]
            
            return Entity(
                text=e1.text if e1.confidence >= e2.confidence else e2.text,
                type=e1.type,
                start=min(e1.start, e2.start),
                end=max(e1.end, e2.end),
                confidence=max(e1.confidence, e2.confidence),
                normalized=e1.normalized,
                metadata=merged_metadata
            )
        
        else:  # MANUAL_REVIEW
            logger.warning(f"Manual review needed for: {e1.text} vs {e2.text}")
            return e1


class BatchProcessor:
    """Handles batch operations for performance"""
    
    def __init__(self, config: GraphBuilderConfig):
        self.config = config
    
    async def process_batches(
        self,
        items: List[Any],
        process_fn,
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: Items to process
            process_fn: Async function to process each batch
            batch_size: Batch size (uses config if not provided)
            
        Returns:
            Combined results from all batches
        """
        batch_size = batch_size or self.config.batch_size
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            
            try:
                batch_results = await process_fn(batch)
                results.extend(batch_results)
                
                if self.config.verbose and i % self.config.log_progress_interval == 0:
                    logger.info(f"Processed {i}/{len(items)} items")
            
            except Exception as e:
                logger.error(f"Batch processing failed at index {i}: {e}")
                
                if self.config.batch_config.retry_failed_batches:
                    # Retry logic
                    for retry in range(self.config.batch_config.max_retries):
                        try:
                            await asyncio.sleep(
                                self.config.batch_config.backoff_factor ** retry
                            )
                            batch_results = await process_fn(batch)
                            results.extend(batch_results)
                            break
                        except Exception as retry_err:
                            if retry == self.config.batch_config.max_retries - 1:
                                logger.error(f"Batch failed after {retry+1} retries: {retry_err}")
        
        return results


class GraphBuilderService:
    """
    Service for building and populating knowledge graph.
    
    This is the main service that orchestrates the graph building process:
    1. Extract entities from documents
    2. Deduplicate and resolve entities
    3. Create nodes in graph (batched)
    4. Extract relationships
    5. Create relationships in graph (batched)
    
    Example:
        ```python
        config = GraphBuilderConfig.high_quality()
        service = GraphBuilderService(
            graph_repo=neo4j_repository,
            entity_extractor=entity_extractor,
            config=config
        )
        
        # Build graph from documents
        result = await service.build_from_documents(documents)
        
        print(f"Created {result.created_nodes} nodes")
        print(f"Created {result.created_relationships} relationships")
        ```
    """
    
    def __init__(
        self,
        graph_repo: GraphRepository,
        entity_extractor: CategoryGuidedEntityExtractor,
        config: Optional[GraphBuilderConfig] = None
    ):
        """
        Initialize GraphBuilderService.
        
        Args:
            graph_repo: Graph repository for database operations
            entity_extractor: Entity extractor for extracting entities from text
            config: Configuration (uses default if not provided)
        """
        self.graph_repo = graph_repo
        self.entity_extractor = entity_extractor
        self.config = config or GraphBuilderConfig.default()
        
        # Initialize components
        self.entity_processor = EntityProcessor(self.config)
        self.relationship_processor = RelationshipProcessor(self.config)
        self.conflict_resolver = ConflictResolver(self.config)
        self.batch_processor = BatchProcessor(self.config)
        
        logger.info(f"GraphBuilderService initialized with config: {self.config.to_dict()}")
    
    async def build_from_documents(
        self,
        documents: List[Document],
        category_hints: Optional[List[NodeCategory]] = None
    ) -> GraphBuildResult:
        """
        Build graph from document collection.
        
        This is the main entry point for graph building.
        
        Args:
            documents: List of documents to process
            category_hints: Optional hints for entity extraction
            
        Returns:
            GraphBuildResult with statistics
        """
        start_time = datetime.now()
        result = GraphBuildResult()
        
        logger.info(f"Building graph from {len(documents)} documents...")
        
        try:
            # Phase 1: Extract all entities
            logger.info("Phase 1: Extracting entities...")
            all_entities = await self._extract_entities(documents, category_hints)
            logger.info(f"  Extracted {len(all_entities)} entities")
            
            # Phase 2: Deduplicate and resolve entities
            logger.info("Phase 2: Deduplicating entities...")
            resolved_entities = self._resolve_entities(all_entities)
            result.deduplicated_entities = len(all_entities) - len(resolved_entities)
            logger.info(f"  Deduplicated: {len(all_entities)} -> {len(resolved_entities)}")
            
            # Phase 3: Create nodes in batches
            logger.info("Phase 3: Creating nodes...")
            node_ids, entity_to_node_id = await self._batch_create_nodes(resolved_entities)
            result.created_nodes = len(node_ids)
            result.failed_nodes = len(resolved_entities) - len(node_ids)
            logger.info(f"  Created {result.created_nodes} nodes")
            
            # Phase 4: Extract and create relationships
            if self.config.extract_relations_from_documents:
                logger.info("Phase 4: Extracting relationships...")
                relationships = await self._extract_relationships(documents, entity_to_node_id)
                logger.info(f"  Extracted {len(relationships)} relationships")
                
                logger.info("Phase 5: Creating relationships...")
                rel_count = await self._batch_create_relationships(relationships)
                result.created_relationships = rel_count
                result.failed_relationships = len(relationships) - rel_count
                logger.info(f"  Created {result.created_relationships} relationships")
            
            # Calculate processing time
            result.processing_time_seconds = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Graph building complete: {result.to_dict()}")
            
        except Exception as e:
            logger.error(f"Graph building failed: {e}")
            result.errors.append(str(e))
            import traceback
            traceback.print_exc()
        
        return result
    
    async def _extract_entities(
        self,
        documents: List[Document],
        category_hints: Optional[List[NodeCategory]] = None
    ) -> List[Entity]:
        """Extract entities from all documents"""
        all_entities = []
        
        for doc in documents:
            try:
                # Extract entities using category-guided extractor
                entities = self.entity_extractor.extract(
                    text=doc.content,
                    categories=[cat.value for cat in category_hints] if category_hints else None
                )
                
                # Validate and normalize entities
                for entity in entities:
                    is_valid, error = self.entity_processor.validate(entity)
                    if is_valid:
                        normalized = self.entity_processor.normalize(entity)
                        all_entities.append(normalized)
                    else:
                        logger.debug(f"Invalid entity rejected: {entity.text} - {error}")
                
            except Exception as e:
                logger.error(f"Entity extraction failed for doc {doc.doc_id}: {e}")
        
        return all_entities
    
    def _resolve_entities(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate and resolve entity references"""
        return self.conflict_resolver.resolve_duplicates(entities)
    
    async def _batch_create_nodes(
        self,
        entities: List[Entity]
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Create nodes in batches for performance.
        
        Returns:
            (list of node IDs, mapping from entity key to node ID)
        """
        node_ids = []
        entity_to_node_id = {}
        
        # Convert entities to nodes
        nodes = []
        entity_keys = []
        
        for entity in entities:
            try:
                node = self.entity_processor.entity_to_node(entity)
                nodes.append(node)
                entity_key = self.relationship_processor._entity_key(entity)
                entity_keys.append(entity_key)
            except Exception as e:
                logger.error(f"Failed to convert entity to node: {entity.text} - {e}")
        
        # Batch create nodes
        async def create_batch(batch_nodes):
            try:
                batch_ids = await self.graph_repo.add_nodes_batch(batch_nodes)
                return batch_ids
            except Exception as e:
                logger.error(f"Batch node creation failed: {e}")
                return []
        
        all_ids = await self.batch_processor.process_batches(
            nodes,
            create_batch,
            self.config.batch_size
        )
        
        # Build mapping
        for i, node_id in enumerate(all_ids):
            if i < len(entity_keys):
                entity_to_node_id[entity_keys[i]] = node_id
                node_ids.append(node_id)
        
        return node_ids, entity_to_node_id
    
    async def _extract_relationships(
        self,
        documents: List[Document],
        entity_to_node_id: Dict[str, str]
    ) -> List[GraphRelationship]:
        """
        Extract relationships from documents.
        
        Week 2: Stub implementation - will be enhanced with LLM extraction
        """
        # TODO: Implement LLM-guided relation extraction (Task B1)
        logger.warning("LLM relation extraction not yet implemented")
        return []
    
    async def _batch_create_relationships(
        self,
        relationships: List[GraphRelationship]
    ) -> int:
        """Create relationships in batches"""
        if not relationships:
            return 0
        
        async def create_batch(batch_rels):
            try:
                count = await self.graph_repo.add_relationships_batch(batch_rels)
                return [count]  # Return as list for consistency
            except Exception as e:
                logger.error(f"Batch relationship creation failed: {e}")
                return [0]
        
        counts = await self.batch_processor.process_batches(
            relationships,
            create_batch,
            self.config.batch_size
        )
        
        return sum(counts)
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get current graph statistics"""
        stats = await self.graph_repo.get_graph_stats()
        return stats
