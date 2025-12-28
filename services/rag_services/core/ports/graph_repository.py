"""
Graph Repository Port (Interface).

This module defines the port (interface) for graph database operations.
Implementations (adapters) will provide concrete Neo4j, Neptune, etc. support.

Following Clean Architecture / Ports & Adapters pattern.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from core.domain.graph_models import (
    GraphNode,
    GraphRelationship,
    GraphPath,
    SubGraph,
    GraphQuery,
    NodeCategory,
    RelationshipType
)


class GraphRepository(ABC):
    """
    Port (interface) for graph database operations.
    
    This abstract base class defines the contract that graph database
    adapters must implement. It provides a clean separation between
    domain logic and infrastructure.
    """
    
    # ========== Node Operations ==========
    
    @abstractmethod
    async def add_node(
        self, 
        node: GraphNode
    ) -> str:
        """
        Add a node to the graph.
        
        Args:
            node: GraphNode domain object
            
        Returns:
            Node ID assigned by the database
            
        Raises:
            DuplicateNodeError: If node with unique property already exists
            ValidationError: If node properties are invalid
        """
        pass
    
    @abstractmethod
    async def get_node(
        self, 
        node_id: str
    ) -> Optional[GraphNode]:
        """
        Retrieve a node by ID.
        
        Args:
            node_id: Unique node identifier
            
        Returns:
            GraphNode if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_nodes_by_category(
        self,
        category: NodeCategory,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[GraphNode]:
        """
        Get nodes of a specific category with optional filters.
        
        Args:
            category: Node category to filter by
            filters: Additional property filters
            limit: Maximum number of nodes to return
            
        Returns:
            List of matching GraphNodes
        """
        pass
    
    @abstractmethod
    async def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any]
    ) -> bool:
        """
        Update node properties.
        
        Args:
            node_id: Node identifier
            properties: Properties to update
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def delete_node(
        self,
        node_id: str,
        cascade: bool = False
    ) -> bool:
        """
        Delete a node from the graph.
        
        Args:
            node_id: Node identifier
            cascade: If True, also delete connected relationships
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    # ========== Relationship Operations ==========
    
    @abstractmethod
    async def add_relationship(
        self,
        relationship: GraphRelationship
    ) -> bool:
        """
        Add a relationship between two nodes.
        
        Args:
            relationship: GraphRelationship domain object
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            NodeNotFoundError: If source or target node doesn't exist
        """
        pass
    
    @abstractmethod
    async def get_relationships(
        self,
        node_id: str,
        rel_type: Optional[RelationshipType] = None,
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> List[GraphRelationship]:
        """
        Get relationships connected to a node.
        
        Args:
            node_id: Node identifier
            rel_type: Filter by relationship type
            direction: Relationship direction
            
        Returns:
            List of GraphRelationships
        """
        pass
    
    @abstractmethod
    async def delete_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: RelationshipType
    ) -> bool:
        """
        Delete a specific relationship.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            rel_type: Relationship type
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    # ========== Graph Traversal Operations (CRITICAL for CatRAG) ==========
    
    @abstractmethod
    async def traverse(
        self,
        start_node_id: str,
        relationship_types: List[RelationshipType],
        max_depth: int = 2,
        direction: str = "outgoing"
    ) -> SubGraph:
        """
        Traverse the graph from a starting node.
        
        CRITICAL for CatRAG: Used for intent-based queries like
        "What are the prerequisites for IT003?"
        
        Args:
            start_node_id: Starting node ID
            relationship_types: Types of relationships to follow
            max_depth: Maximum traversal depth
            direction: Traversal direction
            
        Returns:
            SubGraph containing reachable nodes and relationships
        """
        pass
    
    @abstractmethod
    async def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        max_length: int = 5
    ) -> Optional[GraphPath]:
        """
        Find shortest path between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_types: Types of relationships to follow
            max_length: Maximum path length
            
        Returns:
            GraphPath if path exists, None otherwise
        """
        pass
    
    @abstractmethod
    async def find_all_paths(
        self,
        source_id: str,
        target_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        max_length: int = 3,
        limit: int = 10
    ) -> List[GraphPath]:
        """
        Find all paths between two nodes.
        
        Useful for complex queries like:
        "All ways to satisfy graduation requirements"
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_types: Types of relationships to follow
            max_length: Maximum path length
            limit: Maximum number of paths to return
            
        Returns:
            List of GraphPaths
        """
        pass
    
    @abstractmethod
    async def get_subgraph(
        self,
        center_node_id: str,
        expand_depth: int = 1,
        category_filter: Optional[List[NodeCategory]] = None
    ) -> SubGraph:
        """
        Get a subgraph centered on a node.
        
        Used for context retrieval and visualization.
        
        Args:
            center_node_id: Center node ID
            expand_depth: How many hops to expand
            category_filter: Filter expanded nodes by category
            
        Returns:
            SubGraph
        """
        pass
    
    # ========== Query Operations ==========
    
    @abstractmethod
    async def execute_query(
        self,
        query: GraphQuery
    ) -> SubGraph:
        """
        Execute a graph query.
        
        Args:
            query: GraphQuery domain object
            
        Returns:
            SubGraph with query results
        """
        pass
    
    @abstractmethod
    async def execute_cypher(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute raw Cypher query (Neo4j specific).
        
        For advanced queries not covered by domain operations.
        
        Args:
            cypher: Cypher query string
            params: Query parameters
            
        Returns:
            Query results as list of dicts
        """
        pass
    
    # ========== Bulk Operations ==========
    
    @abstractmethod
    async def add_nodes_batch(
        self,
        nodes: List[GraphNode]
    ) -> List[str]:
        """
        Add multiple nodes in a batch.
        
        More efficient than adding nodes one by one.
        
        Args:
            nodes: List of GraphNodes
            
        Returns:
            List of assigned node IDs
        """
        pass
    
    @abstractmethod
    async def add_relationships_batch(
        self,
        relationships: List[GraphRelationship]
    ) -> int:
        """
        Add multiple relationships in a batch.
        
        Args:
            relationships: List of GraphRelationships
            
        Returns:
            Number of relationships created
        """
        pass
    
    # ========== Search Operations (for Graph-aware search) ==========
    
    @abstractmethod
    async def search_nodes(
        self,
        query: str,
        categories: Optional[List[NodeCategory]] = None,
        limit: int = 10
    ) -> List[GraphNode]:
        """
        Full-text search on nodes.
        
        Uses graph database's built-in text search capabilities.
        
        Args:
            query: Search query string
            categories: Filter by node categories
            limit: Maximum results
            
        Returns:
            List of matching GraphNodes
        """
        pass
    
    # ========== Analytics & Statistics ==========
    
    @abstractmethod
    async def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Returns:
            Dictionary with stats like node counts, relationship counts, etc.
        """
        pass
    
    @abstractmethod
    async def get_category_distribution(self) -> Dict[str, int]:
        """
        Get distribution of nodes by category.
        
        Returns:
            Dictionary mapping category to count
        """
        pass
    
    # ========== Health & Maintenance ==========
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if graph database is healthy and responsive.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def clear_graph(self) -> bool:
        """
        Clear all nodes and relationships (USE WITH CAUTION).
        
        For testing and development only.
        
        Returns:
            True if successful
        """
        pass


# Custom exceptions for graph operations

class GraphRepositoryError(Exception):
    """Base exception for graph repository errors"""
    pass


class NodeNotFoundError(GraphRepositoryError):
    """Raised when a node is not found"""
    pass


class DuplicateNodeError(GraphRepositoryError):
    """Raised when trying to create a duplicate node"""
    pass


class InvalidQueryError(GraphRepositoryError):
    """Raised when query is malformed or invalid"""
    pass


class ConnectionError(GraphRepositoryError):
    """Raised when connection to graph database fails"""
    pass
