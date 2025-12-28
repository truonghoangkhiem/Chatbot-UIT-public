"""
Neo4j Graph Adapter - POC Implementation

This adapter implements the GraphRepository port for Neo4j database.
Provides concrete implementation of graph operations using neo4j-driver.

Part of Clean Architecture - Infrastructure layer.
"""

import logging
from typing import List, Optional, Dict, Any
from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, ConstraintError

from core.ports.graph_repository import (
    GraphRepository,
    NodeNotFoundError,
    DuplicateNodeError,
    InvalidQueryError,
    ConnectionError as GraphConnectionError
)
from core.domain.graph_models import (
    GraphNode,
    GraphRelationship,
    GraphPath,
    SubGraph,
    GraphQuery,
    NodeCategory,
    RelationshipType
)

logger = logging.getLogger(__name__)


class Neo4jGraphAdapter(GraphRepository):
    """
    Neo4j adapter implementing GraphRepository port.
    
    POC Version - Basic functionality for Week 1 demo.
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = None,
        database: str = "neo4j"
    ):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI
            username: Database username
            password: Database password (defaults to NEO4J_PASSWORD env var)
            database: Database name
        """
        import os
        self.uri = uri
        self.username = username
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database
        self._driver = None
        
        logger.info(f"Initializing Neo4j adapter: {uri}")
    
    def _get_driver(self):
        """Get or create Neo4j driver"""
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, self.password)
                )
                # Test connection
                self._driver.verify_connectivity()
                logger.info("✓ Neo4j connection established")
            except ServiceUnavailable as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise GraphConnectionError(f"Cannot connect to Neo4j at {self.uri}") from e
        
        return self._driver
    
    async def add_node(self, node: GraphNode) -> str:
        """
        Add a node to Neo4j.
        
        Example:
            node = create_mon_hoc_node("IT001", "Nhập môn lập trình", 4)
            node_id = await adapter.add_node(node)
        """
        driver = self._get_driver()
        
        # Build Cypher query
        label = node.category.value
        props = node.properties.copy()
        
        # Generate unique ID if not present
        if "id" not in props and "code" not in props:
            import uuid
            props["_generated_id"] = str(uuid.uuid4())
        
        # Create parameterized query
        cypher = f"""
        CREATE (n:{label} $props)
        RETURN elementId(n) as id
        """
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, props=props)
                record = result.single()
                node_id = record["id"]
                
                logger.info(f"✓ Created node: {label} with ID {node_id}")
                return node_id
                
        except ConstraintError as e:
            logger.error(f"Duplicate node constraint violation: {e}")
            raise DuplicateNodeError(f"Node already exists: {props}") from e
        except Exception as e:
            logger.error(f"Error creating node: {e}")
            raise
    
    async def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID"""
        driver = self._get_driver()
        
        cypher = """
        MATCH (n)
        WHERE elementId(n) = $node_id
        RETURN n, labels(n) as labels
        """
        
        with driver.session(database=self.database) as session:
            result = session.run(cypher, node_id=node_id)
            record = result.single()
            
            if not record:
                return None
            
            # Convert to GraphNode
            neo4j_node = record["n"]
            labels = record["labels"]
            
            # Get category from label
            category_label = labels[0]  # Primary label
            try:
                category = NodeCategory(category_label)
            except ValueError:
                logger.warning(f"Unknown category: {category_label}")
                return None
            
            properties = dict(neo4j_node.items())
            
            return GraphNode(
                id=node_id,
                category=category,
                properties=properties
            )
    
    async def add_relationship(self, relationship: GraphRelationship) -> bool:
        """
        Add relationship between nodes.
        
        Example:
            rel = create_prerequisite_relationship(it002_id, it001_id, required=True)
            await adapter.add_relationship(rel)
        """
        driver = self._get_driver()
        
        rel_type = relationship.rel_type.value
        props = relationship.properties
        
        cypher = f"""
        MATCH (source), (target)
        WHERE elementId(source) = $source_id AND elementId(target) = $target_id
        CREATE (source)-[r:{rel_type} $props]->(target)
        RETURN r
        """
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(
                    cypher,
                    source_id=relationship.source_id,
                    target_id=relationship.target_id,
                    props=props
                )
                
                if result.single():
                    logger.info(f"✓ Created relationship: {rel_type}")
                    return True
                else:
                    raise NodeNotFoundError("Source or target node not found")
                    
        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            raise
    
    async def traverse(
        self,
        start_node_id: str,
        relationship_types: List[RelationshipType],
        max_depth: int = 2,
        direction: str = "outgoing"
    ) -> SubGraph:
        """
        Traverse graph from starting node.
        
        CRITICAL for CatRAG queries like "What are prerequisites for IT003?"
        """
        driver = self._get_driver()
        
        # Build relationship pattern
        rel_types_str = "|".join(rt.value for rt in relationship_types)
        
        if direction == "outgoing":
            pattern = f"-[r:{rel_types_str}*1..{max_depth}]->"
        elif direction == "incoming":
            pattern = f"<-[r:{rel_types_str}*1..{max_depth}]-"
        else:  # both
            pattern = f"-[r:{rel_types_str}*1..{max_depth}]-"
        
        cypher = f"""
        MATCH path = (start){pattern}(end)
        WHERE elementId(start) = $start_id
        RETURN path
        """
        
        nodes = []
        relationships = []
        
        with driver.session(database=self.database) as session:
            result = session.run(cypher, start_id=start_node_id)
            
            for record in result:
                path = record["path"]
                
                # Extract nodes
                for neo4j_node in path.nodes:
                    node_id = neo4j_node.element_id
                    labels = list(neo4j_node.labels)
                    
                    if labels:
                        try:
                            category = NodeCategory(labels[0])
                            properties = dict(neo4j_node.items())
                            
                            graph_node = GraphNode(
                                id=node_id,
                                category=category,
                                properties=properties
                            )
                            
                            # Avoid duplicates
                            if not any(n.id == node_id for n in nodes):
                                nodes.append(graph_node)
                        except ValueError:
                            continue
                
                # Extract relationships
                for neo4j_rel in path.relationships:
                    try:
                        rel_type = RelationshipType(neo4j_rel.type)
                        
                        graph_rel = GraphRelationship(
                            source_id=neo4j_rel.start_node.element_id,
                            target_id=neo4j_rel.end_node.element_id,
                            rel_type=rel_type,
                            properties=dict(neo4j_rel.items())
                        )
                        
                        relationships.append(graph_rel)
                    except ValueError:
                        continue
        
        return SubGraph(
            nodes=nodes,
            relationships=relationships,
            center_node_id=start_node_id,
            metadata={
                "max_depth": max_depth,
                "direction": direction,
                "relationship_types": [rt.value for rt in relationship_types]
            }
        )
    
    async def get_nodes_by_category(
        self,
        category: NodeCategory,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[GraphNode]:
        """Get all nodes of a category"""
        driver = self._get_driver()
        
        label = category.value
        
        # Build WHERE clause for filters
        where_clauses = []
        params = {}
        
        if filters:
            for key, value in filters.items():
                where_clauses.append(f"n.{key} = ${key}")
                params[key] = value
        
        where_str = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        cypher = f"""
        MATCH (n:{label})
        {where_str}
        RETURN n, elementId(n) as id
        LIMIT {limit}
        """
        
        nodes = []
        
        with driver.session(database=self.database) as session:
            result = session.run(cypher, **params)
            
            for record in result:
                neo4j_node = record["n"]
                node_id = record["id"]
                properties = dict(neo4j_node.items())
                
                nodes.append(GraphNode(
                    id=node_id,
                    category=category,
                    properties=properties
                ))
        
        return nodes
    
    async def execute_cypher(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute raw Cypher query"""
        driver = self._get_driver()
        params = params or {}
        
        results = []
        
        with driver.session(database=self.database) as session:
            result = session.run(cypher, **params)
            
            for record in result:
                results.append(dict(record))
        
        return results
    
    async def health_check(self) -> bool:
        """Check Neo4j connection health"""
        try:
            driver = self._get_driver()
            with driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as num")
                record = result.single()
                return record["num"] == 1
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics"""
        driver = self._get_driver()
        
        stats = {}
        
        with driver.session(database=self.database) as session:
            # Count nodes by label
            result = session.run("MATCH (n) RETURN labels(n) as labels, count(n) as count")
            node_counts = {}
            for record in result:
                labels = record["labels"]
                if labels:
                    node_counts[labels[0]] = record["count"]
            
            stats["nodes_by_category"] = node_counts
            stats["total_nodes"] = sum(node_counts.values())
            
            # Count relationships by type
            result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
            rel_counts = {}
            for record in result:
                rel_counts[record["type"]] = record["count"]
            
            stats["relationships_by_type"] = rel_counts
            stats["total_relationships"] = sum(rel_counts.values())
        
        return stats
    
    async def clear_graph(self) -> bool:
        """Clear all data (USE WITH CAUTION)"""
        driver = self._get_driver()
        
        logger.warning("⚠️  Clearing entire graph database!")
        
        with driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        logger.info("✓ Graph cleared")
        return True
    
    # ========== Full Implementation - Week 1 Task A4 ==========
    
    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """
        Update node properties.
        
        Args:
            node_id: Neo4j element ID
            properties: Dictionary of properties to update
            
        Returns:
            True if updated successfully
            
        Example:
            await adapter.update_node(node_id, {"so_tin_chi": 5, "updated_at": datetime.now()})
        """
        driver = self._get_driver()
        
        # Build SET clause
        set_clauses = []
        params = {"node_id": node_id}
        
        for key, value in properties.items():
            set_clauses.append(f"n.{key} = ${key}")
            params[key] = value
        
        set_str = ", ".join(set_clauses)
        
        cypher = f"""
        MATCH (n)
        WHERE elementId(n) = $node_id
        SET {set_str}
        RETURN n
        """
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, **params)
                record = result.single()
                
                if record:
                    logger.info(f"✓ Updated node: {node_id}")
                    return True
                else:
                    logger.warning(f"Node not found: {node_id}")
                    return False
        except Exception as e:
            logger.error(f"Error updating node: {e}")
            raise
    
    async def delete_node(self, node_id: str, cascade: bool = False) -> bool:
        """
        Delete node (soft delete with timestamp or hard delete).
        
        Args:
            node_id: Neo4j element ID
            cascade: If True, delete relationships; if False, fail if relationships exist
            
        Returns:
            True if deleted successfully
            
        Example:
            # Soft delete (mark as deleted)
            await adapter.delete_node(node_id, cascade=True)
        """
        driver = self._get_driver()
        
        if cascade:
            # Hard delete with relationships
            cypher = """
            MATCH (n)
            WHERE elementId(n) = $node_id
            DETACH DELETE n
            RETURN count(n) as deleted
            """
        else:
            # Soft delete - set deleted_at timestamp
            cypher = """
            MATCH (n)
            WHERE elementId(n) = $node_id
            SET n.deleted_at = datetime()
            RETURN n
            """
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, node_id=node_id)
                record = result.single()
                
                if record:
                    logger.info(f"✓ Deleted node: {node_id} (cascade={cascade})")
                    return True
                else:
                    logger.warning(f"Node not found: {node_id}")
                    return False
        except Exception as e:
            logger.error(f"Error deleting node: {e}")
            raise
    
    async def get_relationships(
        self, node_id: str, rel_type: Optional[RelationshipType] = None, direction: str = "both"
    ) -> List[GraphRelationship]:
        """
        Get relationships for a node.
        
        Args:
            node_id: Neo4j element ID
            rel_type: Filter by relationship type (optional)
            direction: "outgoing", "incoming", or "both"
            
        Returns:
            List of GraphRelationships
            
        Example:
            rels = await adapter.get_relationships(course_id, RelationshipType.DIEU_KIEN_TIEN_QUYET, "outgoing")
        """
        driver = self._get_driver()
        
        # Build cypher based on direction
        if direction == "outgoing":
            pattern = "(n)-[r]->(m)"
        elif direction == "incoming":
            pattern = "(n)<-[r]-(m)"
        else:  # both
            pattern = "(n)-[r]-(m)"
        
        # Add type filter if specified
        if rel_type:
            type_filter = f":{rel_type.value}"
        else:
            type_filter = ""
        
        cypher = f"""
        MATCH {pattern.replace('[r]', f'[r{type_filter}]')}
        WHERE elementId(n) = $node_id
        RETURN elementId(startNode(r)) as source_id, 
               elementId(endNode(r)) as target_id,
               type(r) as rel_type,
               properties(r) as props
        """
        
        relationships = []
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, node_id=node_id)
                
                for record in result:
                    try:
                        rel_type_value = RelationshipType(record["rel_type"])
                        
                        relationships.append(GraphRelationship(
                            source_id=record["source_id"],
                            target_id=record["target_id"],
                            rel_type=rel_type_value,
                            properties=dict(record["props"])
                        ))
                    except ValueError:
                        logger.warning(f"Unknown relationship type: {record['rel_type']}")
                        continue
        except Exception as e:
            logger.error(f"Error getting relationships: {e}")
            raise
        
        return relationships
    
    async def get_pairs_by_relationship_type(
        self, 
        rel_types: List[str], 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get all node pairs connected by specific relationship types.
        
        This is useful for queries like "List all articles with YEU_CAU or QUY_DINH_DIEU_KIEN relationships"
        
        Args:
            rel_types: List of relationship type names (e.g., ["YEU_CAU", "QUY_DINH_DIEU_KIEN"])
            limit: Maximum number of pairs to return
            
        Returns:
            List of dictionaries containing source and target node information
            
        Example:
            pairs = await adapter.get_pairs_by_relationship_type(["YEU_CAU", "QUY_DINH_DIEU_KIEN"], limit=10)
            # Returns: [{"source": {...}, "target": {...}, "relationship": "YEU_CAU"}, ...]
        """
        driver = self._get_driver()
        
        # Build relationship type pattern
        rel_pattern = "|".join(rel_types)
        
        cypher = f"""
        MATCH (source)-[r:{rel_pattern}]->(target)
        RETURN 
            labels(source)[0] as source_type,
            properties(source) as source_props,
            type(r) as rel_type,
            properties(r) as rel_props,
            labels(target)[0] as target_type,
            properties(target) as target_props
        LIMIT $limit
        """
        
        pairs = []
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, limit=limit)
                
                for record in result:
                    source_props = dict(record["source_props"])
                    target_props = dict(record["target_props"])
                    rel_props = dict(record["rel_props"])
                    
                    # Get display names for source and target
                    # Try various property names in order of preference
                    source_name = (
                        source_props.get("article_id") or
                        source_props.get("title") or
                        source_props.get("name") or
                        source_props.get("ma_mon") or
                        source_props.get("ten_mon") or
                        source_props.get("entity_text") or
                        source_props.get("text") or
                        f"Entity_{list(source_props.values())[0] if source_props else 'Unknown'}"
                    )
                    
                    target_name = (
                        target_props.get("article_id") or
                        target_props.get("title") or
                        target_props.get("name") or
                        target_props.get("ma_mon") or
                        target_props.get("ten_mon") or
                        target_props.get("entity_text") or
                        target_props.get("text") or
                        f"Entity_{list(target_props.values())[0] if target_props else 'Unknown'}"
                    )
                    
                    pairs.append({
                        "source": {
                            "type": record["source_type"],
                            "name": source_name,
                            "properties": source_props
                        },
                        "target": {
                            "type": record["target_type"],
                            "name": target_name,
                            "properties": target_props
                        },
                        "relationship": record["rel_type"],
                        "rel_properties": rel_props
                    })
                    
        except Exception as e:
            logger.error(f"Error getting relationship pairs: {e}")
            raise
        
        return pairs
    
    async def delete_relationship(
        self, source_id: str, target_id: str, rel_type: RelationshipType
    ) -> bool:
        """
        Delete relationship between two nodes.
        
        Args:
            source_id: Source node element ID
            target_id: Target node element ID
            rel_type: Relationship type to delete
            
        Returns:
            True if deleted successfully
            
        Example:
            await adapter.delete_relationship(it003_id, it002_id, RelationshipType.DIEU_KIEN_TIEN_QUYET)
        """
        driver = self._get_driver()
        
        cypher = f"""
        MATCH (source)-[r:{rel_type.value}]->(target)
        WHERE elementId(source) = $source_id AND elementId(target) = $target_id
        DELETE r
        RETURN count(r) as deleted
        """
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, source_id=source_id, target_id=target_id)
                record = result.single()
                
                if record and record["deleted"] > 0:
                    logger.info(f"✓ Deleted relationship: {rel_type.value}")
                    return True
                else:
                    logger.warning(f"Relationship not found")
                    return False
        except Exception as e:
            logger.error(f"Error deleting relationship: {e}")
            raise
    
    async def find_shortest_path(
        self, source_id: str, target_id: str, 
        relationship_types: Optional[List[RelationshipType]] = None, max_length: int = 5
    ) -> Optional[GraphPath]:
        """
        Find shortest path between two nodes (Dijkstra/BFS).
        
        Critical for CatRAG: Finding prerequisite chains for courses.
        
        Args:
            source_id: Source node element ID
            target_id: Target node element ID
            relationship_types: List of relationship types to follow (None = all)
            max_length: Maximum path length
            
        Returns:
            GraphPath if found, None otherwise
            
        Example:
            # Find prerequisite chain from SE363 to IT001
            path = await adapter.find_shortest_path(se363_id, it001_id, [RelationshipType.DIEU_KIEN_TIEN_QUYET])
        """
        driver = self._get_driver()
        
        # Build relationship type filter
        if relationship_types:
            rel_types_str = "|".join([rt.value for rt in relationship_types])
            rel_filter = f":{rel_types_str}"
        else:
            rel_filter = ""
        
        cypher = f"""
        MATCH path = shortestPath((source)-[{rel_filter}*1..{max_length}]->(target))
        WHERE elementId(source) = $source_id AND elementId(target) = $target_id
        RETURN path,
               [node IN nodes(path) | {{id: elementId(node), labels: labels(node), properties: properties(node)}}] as nodes,
               [rel IN relationships(path) | {{type: type(rel), properties: properties(rel)}}] as rels,
               length(path) as path_length
        """
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, source_id=source_id, target_id=target_id)
                record = result.single()
                
                if not record:
                    logger.info(f"No path found between {source_id} and {target_id}")
                    return None
                
                # Convert to GraphPath
                nodes_data = record["nodes"]
                rels_data = record["rels"]
                
                # Build GraphNode objects
                graph_nodes = []
                for node_data in nodes_data:
                    try:
                        category = NodeCategory(node_data["labels"][0])
                        graph_nodes.append(GraphNode(
                            id=node_data["id"],
                            category=category,
                            properties=node_data["properties"]
                        ))
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid node in path")
                        continue
                
                # Build GraphRelationship objects
                graph_rels = []
                for i, rel_data in enumerate(rels_data):
                    try:
                        rel_type = RelationshipType(rel_data["type"])
                        graph_rels.append(GraphRelationship(
                            source_id=nodes_data[i]["id"],
                            target_id=nodes_data[i+1]["id"],
                            rel_type=rel_type,
                            properties=rel_data["properties"]
                        ))
                    except (ValueError, IndexError):
                        logger.warning(f"Invalid relationship in path")
                        continue
                
                return GraphPath(
                    nodes=graph_nodes,
                    relationships=graph_rels
                    # length is auto-calculated
                )
                
        except Exception as e:
            logger.error(f"Error finding shortest path: {e}")
            raise
    
    async def find_all_paths(
        self, source_id: str, target_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        max_length: int = 3, limit: int = 10
    ) -> List[GraphPath]:
        """Find all paths"""
        # TODO: Implement in full version
        raise NotImplementedError("POC: Not yet implemented")
    
    async def get_subgraph(
        self, center_node_id: str, expand_depth: int = 1,
        category_filter: Optional[List[NodeCategory]] = None
    ) -> SubGraph:
        """Get subgraph"""
        # TODO: Implement in full version
        raise NotImplementedError("POC: Not yet implemented")
    
    async def execute_query(self, query: GraphQuery) -> SubGraph:
        """Execute graph query"""
        # TODO: Implement in full version
        raise NotImplementedError("POC: Not yet implemented")
    
    async def add_nodes_batch(self, nodes: List[GraphNode]) -> List[str]:
        """
        Batch add nodes with optimized MERGE query (Deduplication enabled).
        
        Uses MERGE instead of CREATE to prevent duplicate nodes.
        Merges based on the standard ID key for each label (ma_mon, ma_khoa, etc).
        
        Args:
            nodes: List of GraphNodes to add
            
        Returns:
            List of assigned node IDs (elementId from Neo4j)
            
        Example:
            nodes = [create_mon_hoc_node(...), create_mon_hoc_node(...)]
            node_ids = await adapter.add_nodes_batch(nodes)
        """
        if not nodes:
            return []
        
        from core.domain.schema_mapper import SchemaMapper
        
        driver = self._get_driver()
        
        # Group nodes by category for batch processing
        nodes_by_category = {}
        for node in nodes:
            category = node.category.value
            if category not in nodes_by_category:
                nodes_by_category[category] = []
            nodes_by_category[category].append(node)
        
        all_node_ids = []
        
        # Batch insert per category using MERGE
        for category, category_nodes in nodes_by_category.items():
            # Get the standard ID key for this category
            id_key = SchemaMapper.PROPERTY_MAPPING.get(category, {}).get("id_key", "id")
            
            # Prepare data for UNWIND
            nodes_data = []
            for node in category_nodes:
                props = node.properties.copy()
                
                # Ensure ID key exists
                if id_key not in props:
                    # Try to extract from common keys
                    id_value = props.get("code") or props.get("id") or props.get("name")
                    if id_value:
                        # Clean the ID
                        id_value = SchemaMapper.extract_clean_id(str(id_value), category)
                        props[id_key] = id_value
                    else:
                        # Generate fallback ID
                        import uuid
                        props[id_key] = f"auto_{str(uuid.uuid4())[:8]}"
                
                nodes_data.append(props)
            
            # Use MERGE to prevent duplicates
            cypher = f"""
            UNWIND $nodes_data as node_props
            MERGE (n:{category} {{{id_key}: node_props.{id_key}}})
            SET n += node_props
            RETURN elementId(n) as id
            """
            
            try:
                with driver.session(database=self.database) as session:
                    result = session.run(cypher, nodes_data=nodes_data)
                    
                    for record in result:
                        all_node_ids.append(record["id"])
                
                logger.info(f"✓ Batch created/updated {len(category_nodes)} nodes of type {category}")
            except Exception as e:
                logger.error(f"Error in batch node creation: {e}")
                raise
        
        return all_node_ids
    
    async def add_relationships_batch(self, relationships: List[GraphRelationship]) -> int:
        """
        Batch add relationships with optimized UNWIND query.
        
        Args:
            relationships: List of GraphRelationships to add
            
        Returns:
            Number of relationships created
            
        Example:
            rels = [create_prerequisite_relationship(...), ...]
            count = await adapter.add_relationships_batch(rels)
        """
        if not relationships:
            return 0
        
        driver = self._get_driver()
        
        # Group by relationship type
        rels_by_type = {}
        for rel in relationships:
            # Handle both string and enum types
            # Check if it has .value attribute (enum) instead of isinstance
            if hasattr(rel.rel_type, 'value'):
                rel_type = rel.rel_type.value
            else:
                rel_type = rel.rel_type
            
            if rel_type not in rels_by_type:
                rels_by_type[rel_type] = []
            rels_by_type[rel_type].append(rel)
        
        total_created = 0
        
        # Batch insert per type
        for rel_type, type_rels in rels_by_type.items():
            # Prepare data for UNWIND
            rels_data = []
            for rel in type_rels:
                rels_data.append({
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "props": rel.properties
                })
            
            cypher = f"""
            UNWIND $rels_data as rel_data
            MATCH (source), (target)
            WHERE elementId(source) = rel_data.source_id AND elementId(target) = rel_data.target_id
            CREATE (source)-[r:{rel_type}]->(target)
            SET r = rel_data.props
            RETURN count(r) as created
            """
            
            try:
                with driver.session(database=self.database) as session:
                    result = session.run(cypher, rels_data=rels_data)
                    record = result.single()
                    created = record["created"] if record else 0
                    total_created += created
                
                logger.info(f"✓ Batch created {created} relationships of type {rel_type}")
            except Exception as e:
                logger.error(f"Error in batch relationship creation: {e}")
                raise
        
        return total_created
    
    async def search_nodes(
        self, query: str, categories: Optional[List[NodeCategory]] = None, limit: int = 10
    ) -> List[GraphNode]:
        """
        Full-text search on nodes using Neo4j's full-text indexes.
        
        Uses the full-text indexes created in 02_create_indexes.cypher.
        
        Args:
            query: Search query string (Vietnamese or English)
            categories: Filter by node categories (optional)
            limit: Maximum results
            
        Returns:
            List of matching GraphNodes sorted by relevance score
            
        Example:
            # Search for courses about "cấu trúc dữ liệu"
            results = await adapter.search_nodes("cấu trúc dữ liệu", [NodeCategory.MON_HOC], limit=5)
        """
        driver = self._get_driver()
        
        # Map categories to index names
        index_names = {
            NodeCategory.MON_HOC: "mon_hoc_fulltext",
            NodeCategory.KHOA: "khoa_fulltext",
            NodeCategory.CHUONG_TRINH_DAO_TAO: "chuong_trinh_fulltext",
            NodeCategory.QUY_DINH: "quy_dinh_fulltext",
            NodeCategory.GIANG_VIEN: "giang_vien_fulltext"
        }
        
        all_results = []
        
        # Search in specified categories or all
        search_categories = categories if categories else list(index_names.keys())
        
        for category in search_categories:
            index_name = index_names.get(category)
            if not index_name:
                continue
            
            cypher = """
            CALL db.index.fulltext.queryNodes($index_name, $search_text)
            YIELD node, score
            RETURN elementId(node) as id, node, score
            ORDER BY score DESC
            LIMIT $limit
            """
            
            try:
                with driver.session(database=self.database) as session:
                    result = session.run(cypher, index_name=index_name, search_text=query, limit=limit)
                    
                    for record in result:
                        neo4j_node = record["node"]
                        node_id = record["id"]
                        score = record["score"]
                        
                        properties = dict(neo4j_node.items())
                        properties["_search_score"] = score  # Add relevance score
                        
                        graph_node = GraphNode(
                            id=node_id,
                            category=category,
                            properties=properties
                        )
                        
                        all_results.append((score, graph_node))
            except Exception as e:
                logger.warning(f"Full-text search failed for {category}: {e}")
                continue
        
        # Sort by score and return nodes
        all_results.sort(key=lambda x: x[0], reverse=True)
        return [node for score, node in all_results[:limit]]
    
    # ========== CatRAG-Specific Methods ==========
    
    async def find_prerequisites_chain(
        self, 
        course_code: str,
        max_depth: int = 10
    ) -> List[GraphPath]:
        """
        Find all prerequisite chains for a course (CatRAG-specific).
        
        This is a critical query for the Router Agent to handle prerequisite questions.
        Returns all paths from the course to foundational courses.
        
        Args:
            course_code: Course code (e.g., "SE363", "IT004")
            max_depth: Maximum prerequisite chain depth
            
        Returns:
            List of GraphPaths representing prerequisite chains
            
        Example:
            # Find all prerequisites for SE363 (AI course)
            chains = await adapter.find_prerequisites_chain("SE363")
            # Returns: SE363 -> IT003 -> IT002 -> IT001
        """
        driver = self._get_driver()
        
        cypher = f"""
        MATCH (target:MON_HOC {{ma_mon: $course_code}})
        MATCH path = (target)-[:DIEU_KIEN_TIEN_QUYET*1..{max_depth}]->(prereq:MON_HOC)
        WITH path, length(path) as depth
        ORDER BY depth DESC
        RETURN path,
               [node IN nodes(path) | {{
                   id: elementId(node),
                   code: node.ma_mon,
                   name: node.ten_mon,
                   credits: node.so_tin_chi
               }}] as nodes,
               [rel IN relationships(path) | {{
                   type: type(rel),
                   required: rel.loai = 'bat_buoc',
                   min_grade: rel.diem_toi_thieu
               }}] as rels,
               depth
        """
        
        paths = []
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, course_code=course_code)
                
                for record in result:
                    nodes_data = record["nodes"]
                    rels_data = record["rels"]
                    depth = record["depth"]
                    
                    # Build GraphNode objects
                    graph_nodes = []
                    for node_data in nodes_data:
                        graph_nodes.append(GraphNode(
                            id=node_data["id"],
                            category=NodeCategory.MON_HOC,
                            properties={
                                "code": node_data["code"],  # Required by validation
                                "ma_mon": node_data["code"],
                                "name": node_data["name"],  # Required
                                "ten_mon": node_data["name"],
                                "credits": node_data["credits"],  # Required
                                "so_tin_chi": node_data["credits"]
                            }
                        ))
                    
                    # Build GraphRelationship objects
                    graph_rels = []
                    for i, rel_data in enumerate(rels_data):
                        graph_rels.append(GraphRelationship(
                            source_id=nodes_data[i]["id"],
                            target_id=nodes_data[i+1]["id"],
                            rel_type=RelationshipType.DIEU_KIEN_TIEN_QUYET,
                            properties={
                                "loai": "bat_buoc" if rel_data["required"] else "khuyen_nghi",
                                "diem_toi_thieu": rel_data["min_grade"]
                            }
                        ))
                    
                    paths.append(GraphPath(
                        nodes=graph_nodes,
                        relationships=graph_rels
                        # length is auto-calculated in __post_init__
                    ))
            
            logger.info(f"Found {len(paths)} prerequisite paths for {course_code}")
            return paths
            
        except Exception as e:
            logger.error(f"Error finding prerequisite chain: {e}")
            raise
    
    async def find_related_courses(
        self,
        course_code: str,
        similarity_threshold: float = 0.7,
        limit: int = 5
    ) -> List[GraphNode]:
        """
        Find semantically related courses (CatRAG-specific).
        
        Uses LIEN_QUAN relationships with similarity scores.
        
        Args:
            course_code: Course code
            similarity_threshold: Minimum similarity score (0.0-1.0)
            limit: Maximum results
            
        Returns:
            List of related course GraphNodes
        """
        driver = self._get_driver()
        
        cypher = """
        MATCH (source:MON_HOC {ma_mon: $course_code})-[r:LIEN_QUAN]-(related:MON_HOC)
        WHERE r.do_tuong_tu >= $threshold
        RETURN elementId(related) as id, related, r.do_tuong_tu as similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        related_courses = []
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(
                    cypher,
                    course_code=course_code,
                    threshold=similarity_threshold,
                    limit=limit
                )
                
                for record in result:
                    neo4j_node = record["related"]
                    node_id = record["id"]
                    similarity = record["similarity"]
                    
                    properties = dict(neo4j_node.items())
                    properties["_similarity_score"] = similarity
                    
                    related_courses.append(GraphNode(
                        id=node_id,
                        category=NodeCategory.MON_HOC,
                        properties=properties
                    ))
            
            logger.info(f"Found {len(related_courses)} related courses for {course_code}")
            return related_courses
            
        except Exception as e:
            logger.error(f"Error finding related courses: {e}")
            raise
    
    async def get_category_distribution(self) -> Dict[str, int]:
        """Get category distribution"""
        stats = await self.get_graph_stats()
        return stats.get("nodes_by_category", {})
    
    # ========== CatRAG Schema Methods (Article, Entity, Community) ==========
    
    async def search_articles_by_keyword(
        self, 
        keywords: List[str], 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search Article nodes by keywords in title and content.
        
        For CatRAG schema with Article nodes containing 'title', 'content' fields.
        
        ENHANCED: 
        - Prioritize compound terms (phrases with spaces) by requiring exact match
        - Use scoring to rank results by number of keywords matched
        """
        logger.info(f"🔍 search_articles_by_keyword called with keywords: {keywords}, limit: {limit}")
        driver = self._get_driver()
        
        # Separate compound terms (phrases with spaces) from single keywords
        compound_terms = [kw for kw in keywords if ' ' in kw]
        single_keywords = [kw for kw in keywords if ' ' not in kw]
        
        logger.info(f"   Compound terms: {compound_terms}")
        logger.info(f"   Single keywords: {single_keywords}")
        
        # Build search pattern
        # For compound terms: require exact match (AND logic within the compound)
        # For single keywords: use OR logic
        conditions = []
        params = {"limit": limit}
        
        # Compound terms get higher priority - if any compound term matches, include
        for i, ct in enumerate(compound_terms):
            param_name = f"ct{i}"
            conditions.append(
                f"(toLower(a.title) CONTAINS toLower(${param_name}) OR toLower(a.full_text) CONTAINS toLower(${param_name}))"
            )
            params[param_name] = ct
        
        # Single keywords as fallback
        for i, kw in enumerate(single_keywords):
            param_name = f"kw{i}"
            conditions.append(
                f"(toLower(a.title) CONTAINS toLower(${param_name}) OR toLower(a.full_text) CONTAINS toLower(${param_name}))"
            )
            params[param_name] = kw
        
        where_clause = " OR ".join(conditions) if conditions else "true"
        
        # ENHANCED: Add scoring to prioritize articles matching more keywords
        # and especially compound terms
        score_parts = []
        for i in range(len(compound_terms)):
            param_name = f"ct{i}"
            # Compound terms get weight 3 (higher priority)
            score_parts.append(f"CASE WHEN toLower(a.title) CONTAINS toLower(${param_name}) THEN 3 ELSE 0 END")
            score_parts.append(f"CASE WHEN toLower(a.full_text) CONTAINS toLower(${param_name}) THEN 2 ELSE 0 END")
        
        for i in range(len(single_keywords)):
            param_name = f"kw{i}"
            # Single keywords get weight 1
            score_parts.append(f"CASE WHEN toLower(a.title) CONTAINS toLower(${param_name}) THEN 1 ELSE 0 END")
            score_parts.append(f"CASE WHEN toLower(a.full_text) CONTAINS toLower(${param_name}) THEN 0.5 ELSE 0 END")
        
        score_expr = " + ".join(score_parts) if score_parts else "0"
        
        cypher = f"""
        MATCH (a:Article)
        WHERE {where_clause}
        WITH a, ({score_expr}) as score
        RETURN elementId(a) as element_id, a.id as article_id, a.title as title, 
               a.full_text as content, score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        results = []
        try:
            logger.info(f"📝 Executing Cypher with scoring...")
            logger.info(f"📝 With params: {list(params.keys())}")
            with driver.session(database=self.database) as session:
                result = session.run(cypher, **params)
                for record in result:
                    results.append({
                        "id": record["element_id"],
                        "article_id": record["article_id"],
                        "title": record["title"],
                        "content": record["content"][:500] if record["content"] else "",
                        "type": "Article",
                        "score": record["score"]
                    })
            logger.info(f"✅ Found {len(results)} articles for keywords: {keywords}")
            if results:
                logger.info(f"   Top result: {results[0].get('title')} (score: {results[0].get('score')})")
        except Exception as e:
            logger.error(f"❌ Error searching articles: {e}", exc_info=True)
        
        return results
    
    async def search_entities_by_keyword(
        self, 
        keywords: List[str], 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search Entity nodes by name.
        
        For CatRAG schema with Entity nodes containing 'name', 'type', 'description' fields.
        """
        driver = self._get_driver()
        
        # Build search pattern
        keyword_conditions = []
        for i, kw in enumerate(keywords):
            keyword_conditions.append(
                f"(toLower(e.name) CONTAINS toLower($kw{i}) OR toLower(e.description) CONTAINS toLower($kw{i}))"
            )
        
        where_clause = " OR ".join(keyword_conditions) if keyword_conditions else "true"
        
        cypher = f"""
        MATCH (e:Entity)
        WHERE {where_clause}
        RETURN elementId(e) as id, e.name as name, e.type as type, 
               e.description as description
        LIMIT $limit
        """
        
        params = {"limit": limit}
        for i, kw in enumerate(keywords):
            params[f"kw{i}"] = kw
        
        results = []
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, **params)
                for record in result:
                    results.append({
                        "id": record["id"],
                        "name": record["name"],
                        "type": record["type"],
                        "description": record["description"][:200] if record["description"] else "",
                        "node_type": "Entity"
                    })
            logger.info(f"Found {len(results)} entities for keywords: {keywords}")
        except Exception as e:
            logger.error(f"Error searching entities: {e}")
        
        return results
    
    async def get_article_with_entities(
        self, 
        article_number: int
    ) -> Dict[str, Any]:
        """
        Get an Article by its article_number along with related entities.
        
        Returns:
            Dict with article info and list of entities it MENTIONS
        """
        driver = self._get_driver()
        
        # Convert article_number to article id (e.g., 19 -> "dieu_19")
        article_id = f"dieu_{article_number}"
        
        cypher = """
        MATCH (a:Article {id: $article_id})
        OPTIONAL MATCH (a)-[:MENTIONS]->(e:Entity)
        RETURN a.title as title, a.full_text as content,
               collect(DISTINCT {name: e.text, type: e.type}) as entities
        """
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, article_id=article_id)
                record = result.single()
                
                if record:
                    # Filter out null entities
                    entities = [e for e in record["entities"] if e.get("name")]
                    return {
                        "title": record["title"],
                        "content": record["content"],
                        "article_number": article_number,
                        "entities": entities,
                        "type": "Article"
                    }
        except Exception as e:
            logger.error(f"Error getting article {article_number}: {e}")
        
        return {}
    
    async def get_all_communities(self) -> List[Dict[str, Any]]:
        """
        Get all Community nodes with their summaries.
        
        For GLOBAL reasoning - returns community summaries.
        """
        driver = self._get_driver()
        
        cypher = """
        MATCH (c:Community)
        OPTIONAL MATCH (a:Article)-[:BELONGS_TO]->(c)
        RETURN c.name as name, c.summary as summary, c.size as size,
               c.key_entities as key_entities,
               collect(DISTINCT a.title) as articles
        ORDER BY c.size DESC
        """
        
        results = []
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher)
                for record in result:
                    results.append({
                        "name": record["name"],
                        "summary": record["summary"],
                        "size": record["size"],
                        "key_entities": record["key_entities"],
                        "articles": record["articles"][:5],  # Limit article titles
                        "type": "Community"
                    })
            logger.info(f"Found {len(results)} communities")
        except Exception as e:
            logger.error(f"Error getting communities: {e}")
        
        return results
    
    async def find_article_path(
        self, 
        start_article_number: int, 
        end_article_number: int,
        max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find path between two Articles via NEXT_ARTICLE or shared Entity relationships.
        
        For MULTI_HOP reasoning.
        """
        driver = self._get_driver()
        
        # Try NEXT_ARTICLE path first
        cypher = """
        MATCH (start:Article {article_number: $start_num})
        MATCH (end:Article {article_number: $end_num})
        MATCH path = shortestPath((start)-[:NEXT_ARTICLE*..%d]->(end))
        RETURN [node in nodes(path) | {
            article_number: node.article_number,
            title: node.title
        }] as path_nodes,
        length(path) as path_length
        """ % max_depth
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(
                    cypher, 
                    start_num=start_article_number, 
                    end_num=end_article_number
                )
                record = result.single()
                
                if record:
                    return [{
                        "path_nodes": record["path_nodes"],
                        "path_length": record["path_length"],
                        "path_type": "NEXT_ARTICLE"
                    }]
        except Exception as e:
            logger.warning(f"No NEXT_ARTICLE path found: {e}")
        
        # Fallback: Try via shared entities
        cypher2 = """
        MATCH (start:Article {article_number: $start_num})
        MATCH (end:Article {article_number: $end_num})
        MATCH path = (start)-[:MENTIONS]->(:Entity)<-[:MENTIONS]-(end)
        RETURN [node in nodes(path) | {
            name: coalesce(node.title, node.name),
            type: labels(node)[0]
        }] as path_nodes,
        length(path) as path_length
        LIMIT 3
        """
        
        results = []
        try:
            with driver.session(database=self.database) as session:
                result = session.run(
                    cypher2, 
                    start_num=start_article_number, 
                    end_num=end_article_number
                )
                for record in result:
                    results.append({
                        "path_nodes": record["path_nodes"],
                        "path_length": record["path_length"],
                        "path_type": "SHARED_ENTITY"
                    })
        except Exception as e:
            logger.error(f"Error finding article path: {e}")
        
        return results
    
    async def get_related_articles_by_entity(
        self, 
        entity_name: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find Articles that MENTION a specific Entity.
        
        For LOCAL reasoning - "Các điều khoản nào liên quan đến X?"
        """
        driver = self._get_driver()
        
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($entity_name)
        WITH e LIMIT 1
        MATCH (a:Article)-[:MENTIONS]->(e)
        RETURN a.article_number as article_number, a.title as title,
               a.content as content, e.name as entity_name
        LIMIT $limit
        """
        
        results = []
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, entity_name=entity_name, limit=limit)
                for record in result:
                    results.append({
                        "article_number": record["article_number"],
                        "title": record["title"],
                        "content": record["content"][:500] if record["content"] else "",
                        "related_entity": record["entity_name"],
                        "type": "Article"
                    })
            logger.info(f"Found {len(results)} articles mentioning '{entity_name}'")
        except Exception as e:
            logger.error(f"Error finding related articles: {e}")
        
        return results

    async def get_latest_version_of_article(
        self,
        article_title: str
    ) -> Dict[str, Any]:
        """
        Find the latest version of an article by checking AMENDS relationships.
        
        If Article A AMENDS Article B, then A is the newer version.
        Returns the amending article (source of AMENDS) with the original article info.
        
        Use case: When user asks about "Điều 14", check if it has been amended
        and return the newer content along with amendment info.
        
        Args:
            article_title: Title or partial title of the article to check
            
        Returns:
            Dict with:
            - original_article: The article that was searched for
            - amending_article: The article that amends it (if any)
            - amendment_description: Description of the amendment
            - is_amended: Boolean indicating if article has been amended
        """
        driver = self._get_driver()
        
        # First find the article being queried
        # Then check if any article AMENDS it
        cypher = """
        MATCH (original:Article)
        WHERE toLower(original.title) CONTAINS toLower($title_search)
        OPTIONAL MATCH (amending:Article)-[r:AMENDS]->(original)
        RETURN original.title as original_title,
               original.full_text as original_text,
               amending.title as amending_title,
               amending.full_text as amending_text,
               r.description as amendment_description
        LIMIT 5
        """
        
        results = {
            "original_article": None,
            "amending_article": None,
            "amendment_description": None,
            "is_amended": False,
            "all_amendments": []
        }
        
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, title_search=article_title)
                
                for record in result:
                    # Set original article info (first match)
                    if results["original_article"] is None:
                        results["original_article"] = {
                            "title": record["original_title"],
                            "text": record["original_text"]
                        }
                    
                    # Check for amendments
                    if record["amending_title"]:
                        results["is_amended"] = True
                        
                        # Extract ONLY the relevant section for this article
                        amending_text = record["amending_text"] or ""
                        original_title = record["original_title"] or article_title
                        
                        extracted_section = self._extract_amendment_section(
                            amending_text, 
                            original_title
                        )
                        
                        amendment_info = {
                            "amending_title": record["amending_title"],
                            "amending_text": extracted_section or amending_text,
                            "description": record["amendment_description"],
                            "extracted": extracted_section is not None
                        }
                        results["all_amendments"].append(amendment_info)
                        
                        # Set primary amending article (first one found)
                        if results["amending_article"] is None:
                            results["amending_article"] = {
                                "title": record["amending_title"],
                                "text": extracted_section or amending_text,
                                "extracted": extracted_section is not None
                            }
                            results["amendment_description"] = record["amendment_description"]
                
                if results["is_amended"]:
                    logger.info(f"Found {len(results['all_amendments'])} amendments for '{article_title}'")
                else:
                    logger.info(f"No amendments found for '{article_title}'")
                    
        except Exception as e:
            logger.error(f"Error finding article amendments: {e}")
        
        return results

    async def get_all_amendments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get all AMENDS relationships in the graph.
        
        Returns list of amendment info for displaying to user or debugging.
        """
        driver = self._get_driver()
        
        cypher = """
        MATCH (amending:Article)-[r:AMENDS]->(original:Article)
        RETURN amending.title as amending_title,
               original.title as original_title,
               r.description as description,
               amending.full_text as amending_text
        LIMIT $limit
        """
        
        results = []
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, limit=limit)
                for record in result:
                    results.append({
                        "amending_title": record["amending_title"],
                        "original_title": record["original_title"],
                        "description": record["description"],
                        "amending_text": record["amending_text"][:500] if record["amending_text"] else None
                    })
            logger.info(f"Found {len(results)} AMENDS relationships")
        except Exception as e:
            logger.error(f"Error getting amendments: {e}")
        
        return results

    async def search_with_amendments(
        self,
        keywords: List[str],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search articles by keywords and automatically include amendment info.
        
        This is the KEY method for ensuring chatbot answers with latest info.
        When an article is found that has been amended, include BOTH:
        1. The original article (for context)
        2. The amending article with updated content (marked as priority)
        
        Args:
            keywords: List of search keywords
            limit: Max results
            
        Returns:
            List of articles with amendment enrichment
        """
        driver = self._get_driver()
        
        # Search for articles and include any amendments
        # Priority: title match > full_text match
        cypher = """
        // First find articles matching keywords
        MATCH (a:Article)
        WHERE any(kw IN $keywords WHERE 
            toLower(a.full_text) CONTAINS toLower(kw) OR
            toLower(a.title) CONTAINS toLower(kw)
        )
        
        // Calculate relevance score: title match gets higher priority
        WITH a,
            REDUCE(score = 0, kw IN $keywords | 
                score + CASE WHEN toLower(a.title) CONTAINS toLower(kw) THEN 10 ELSE 0 END +
                CASE WHEN toLower(a.full_text) CONTAINS toLower(kw) THEN 1 ELSE 0 END
            ) as relevance
        
        // Check if this article has been amended
        OPTIONAL MATCH (amending:Article)-[r:AMENDS]->(a)
        
        // Return both original and amendment info, sorted by relevance
        RETURN a.title as title,
               a.full_text as text,
               a.id as id,
               CASE WHEN amending IS NOT NULL THEN true ELSE false END as is_amended,
               amending.title as amending_title,
               amending.full_text as amending_text,
               r.description as amendment_description,
               relevance
        ORDER BY relevance DESC
        LIMIT $limit
        """
        
        results = []
        try:
            with driver.session(database=self.database) as session:
                result = session.run(cypher, keywords=keywords, limit=limit)
                
                for record in result:
                    article = {
                        "title": record["title"],
                        "text": record["text"],
                        "id": record["id"],
                        "is_amended": record["is_amended"],
                        "type": "Article"
                    }
                    
                    # If article has been amended, extract ONLY the relevant section
                    if record["is_amended"] and record["amending_text"]:
                        original_title = record["title"] or ""
                        amending_text = record["amending_text"] or ""
                        
                        # Extract the specific section for THIS article from the amending document
                        extracted_section = self._extract_amendment_section(
                            amending_text, 
                            original_title
                        )
                        
                        if extracted_section:
                            # Use extracted section as the new content
                            article["text"] = extracted_section
                            article["is_amended"] = True
                            article["_replaced_from"] = original_title
                        else:
                            # Fallback: keep original but mark as amended
                            article["is_amended"] = True
                            article["amendment_note"] = record["amendment_description"]
                    
                    results.append(article)
                    
            logger.info(f"Found {len(results)} articles for keywords {keywords}")
            
        except Exception as e:
            logger.error(f"Error searching with amendments: {e}")
        
        return results

    def _extract_amendment_section(
        self, 
        amending_text: str, 
        original_title: str
    ) -> Optional[str]:
        """
        Extract the specific amendment section from the amending document
        that corresponds to the original article.
        
        For example, if original_title is "Điều 14" and amending_text contains:
        "--- Mục b khoản 1 Điều 14 ---\nContent here...\n--- Điều 23 ---"
        
        This method extracts ONLY the content between those markers.
        
        Args:
            amending_text: Full text of the amending document (e.g., Điều 1)
            original_title: Title of the original article being amended (e.g., "Điều 14")
            
        Returns:
            The extracted section text, or None if not found
        """
        import re
        
        if not amending_text or not original_title:
            return None
            
        # Extract the article number from title (e.g., "Điều 14" -> "14")
        article_match = re.search(r'Điều\s*(\d+)', original_title, re.IGNORECASE)
        if not article_match:
            return None
            
        article_number = article_match.group(1)
        
        # Pattern to find section markers like:
        # "--- Khoản X Điều Y ---" or "--- Mục X khoản Y Điều Z ---"
        # The content follows until the next "---" marker or end of text
        
        # Build pattern to find this article's section
        # Match variations like:
        # - "--- Điều 14 ---"
        # - "--- Khoản 1 Điều 14 ---"
        # - "--- Mục b khoản 1 Điều 14 ---"
        section_pattern = rf'---[^-]*Điều\s*{article_number}\s*---\s*(.*?)(?=---|\Z)'
        
        match = re.search(section_pattern, amending_text, re.DOTALL | re.IGNORECASE)
        
        if match:
            extracted = match.group(1).strip()
            if extracted:
                logger.info(f"Extracted amendment section for {original_title}: {len(extracted)} chars")
                return extracted
        
        # Alternative: Try finding content that mentions the article directly
        # This handles cases where formatting is different
        alt_pattern = rf'(?:sửa đổi|thay thế|bổ sung)[^.]*Điều\s*{article_number}[^:]*:\s*(.*?)(?=(?:sửa đổi|thay thế|bổ sung)[^.]*Điều|\Z)'
        
        alt_match = re.search(alt_pattern, amending_text, re.DOTALL | re.IGNORECASE)
        
        if alt_match:
            extracted = alt_match.group(1).strip()
            if extracted:
                logger.info(f"Extracted amendment (alt pattern) for {original_title}: {len(extracted)} chars")
                return extracted
        
        logger.warning(f"Could not extract specific section for {original_title}")
        return None

    def close(self):
        """Close Neo4j driver"""
        if self._driver:
            self._driver.close()
            logger.info("✓ Neo4j connection closed")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()


# Factory function for easy instantiation
def create_neo4j_adapter(
    uri: str = "bolt://localhost:7687",
    username: str = "neo4j",
    password: str = None
) -> Neo4jGraphAdapter:
    """
    Factory function to create Neo4j adapter.
    
    Example:
        adapter = create_neo4j_adapter()
        await adapter.health_check()
    """
    import os
    password = password or os.getenv("NEO4J_PASSWORD", "password")
    return Neo4jGraphAdapter(uri=uri, username=username, password=password)
