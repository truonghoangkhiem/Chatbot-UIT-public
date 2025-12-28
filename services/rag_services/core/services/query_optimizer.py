"""
Query Optimizer for Graph Database Operations

Optimizes Cypher queries for Neo4j graph database with features:
- Query plan analysis and optimization
- Query result caching with TTL
- Index usage recommendations
- Cost estimation for complex queries
- Query rewriting for better performance
- Statistics tracking and monitoring

Author: GitHub Copilot
Date: November 2025
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import time
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Query optimization levels"""
    NONE = "none"  # No optimization
    BASIC = "basic"  # Basic optimizations only
    AGGRESSIVE = "aggressive"  # All optimizations


class CacheStrategy(Enum):
    """Query result caching strategies"""
    NO_CACHE = "no_cache"
    LRU = "lru"  # Least Recently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on query pattern


@dataclass
class QueryPlan:
    """Represents a query execution plan"""
    query: str
    estimated_cost: float
    estimated_rows: int
    uses_index: bool
    index_hits: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)
    cache_key: Optional[str] = None


@dataclass
class CachedResult:
    """Cached query result with metadata"""
    result: Any
    query_hash: str
    timestamp: datetime
    ttl_seconds: int
    access_count: int = 0
    last_access: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if cached result is expired"""
        if self.ttl_seconds <= 0:
            return False
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl_seconds
    
    def is_stale(self, staleness_threshold: int = 3600) -> bool:
        """Check if result is stale (old but not expired)"""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > staleness_threshold


@dataclass
class QueryStatistics:
    """Query execution statistics"""
    query_hash: str
    execution_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    last_execution: Optional[datetime] = None
    
    def update(self, execution_time: float):
        """Update statistics with new execution"""
        self.execution_count += 1
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.execution_count
        self.last_execution = datetime.now()


class QueryOptimizer:
    """
    Optimizes Cypher queries for Neo4j graph database
    
    Features:
    - Query result caching with configurable TTL
    - Query plan analysis and cost estimation
    - Index usage recommendations
    - Query rewriting for performance
    - Statistics tracking
    
    Example:
        optimizer = QueryOptimizer(
            enable_cache=True,
            cache_ttl=300,
            optimization_level=OptimizationLevel.AGGRESSIVE
        )
        
        # Analyze query before execution
        plan = optimizer.analyze_query(cypher_query)
        print(f"Estimated cost: {plan.estimated_cost}")
        
        # Execute with caching
        result = await optimizer.execute_cached(session, cypher_query, params)
    """
    
    def __init__(
        self,
        enable_cache: bool = True,
        cache_ttl: int = 300,  # 5 minutes default
        max_cache_size: int = 1000,
        cache_strategy: CacheStrategy = CacheStrategy.TTL,
        optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
        enable_query_rewriting: bool = True,
    ):
        """
        Initialize Query Optimizer
        
        Args:
            enable_cache: Enable query result caching
            cache_ttl: Time to live for cached results (seconds)
            max_cache_size: Maximum number of cached results
            cache_strategy: Caching strategy to use
            optimization_level: Level of query optimization
            enable_query_rewriting: Enable automatic query rewriting
        """
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        self.cache_strategy = cache_strategy
        self.optimization_level = optimization_level
        self.enable_query_rewriting = enable_query_rewriting
        
        # Cache storage
        self._cache: Dict[str, CachedResult] = {}
        self._cache_access_order: List[str] = []  # For LRU
        
        # Statistics tracking
        self._query_stats: Dict[str, QueryStatistics] = defaultdict(QueryStatistics)
        
        # Index tracking (populated from database)
        self._known_indexes: Set[str] = set()
        
        logger.info(
            f"QueryOptimizer initialized: cache={enable_cache}, "
            f"ttl={cache_ttl}s, level={optimization_level.value}"
        )
    
    def analyze_query(self, query: str, params: Optional[Dict] = None) -> QueryPlan:
        """
        Analyze query and provide optimization plan
        
        Args:
            query: Cypher query string
            params: Query parameters
            
        Returns:
            QueryPlan with cost estimates and suggestions
        """
        plan = QueryPlan(
            query=query,
            estimated_cost=0.0,
            estimated_rows=0,
            uses_index=False,
        )
        
        # Generate cache key
        plan.cache_key = self._generate_cache_key(query, params)
        
        # Check for MATCH patterns
        if "MATCH" in query.upper():
            plan.estimated_cost += 10.0
            
            # Check if using WHERE clause (better than no filter)
            if "WHERE" in query.upper():
                plan.estimated_cost *= 0.5
            else:
                plan.optimization_suggestions.append(
                    "Consider adding WHERE clause to filter results early"
                )
        
        # Check for index usage hints
        if any(idx in query for idx in self._known_indexes):
            plan.uses_index = True
            plan.estimated_cost *= 0.1  # Significant cost reduction
        else:
            # Suggest index creation
            if "WHERE" in query.upper():
                properties = self._extract_where_properties(query)
                for prop in properties:
                    plan.optimization_suggestions.append(
                        f"Consider creating index on property: {prop}"
                    )
        
        # Check for expensive operations
        if "OPTIONAL MATCH" in query.upper():
            plan.estimated_cost *= 2.0
            plan.optimization_suggestions.append(
                "OPTIONAL MATCH can be expensive, consider if necessary"
            )
        
        if "ORDER BY" in query.upper() and "LIMIT" not in query.upper():
            plan.optimization_suggestions.append(
                "ORDER BY without LIMIT sorts entire result set"
            )
        
        # Check for DISTINCT usage
        if "DISTINCT" in query.upper():
            plan.estimated_cost *= 1.5
        
        # Estimate row count based on patterns
        if "LIMIT" in query.upper():
            try:
                limit_value = int(query.upper().split("LIMIT")[1].strip().split()[0])
                plan.estimated_rows = limit_value
            except:
                plan.estimated_rows = 100
        else:
            plan.estimated_rows = 1000  # Default estimate
        
        logger.debug(f"Query analysis: cost={plan.estimated_cost:.2f}, rows={plan.estimated_rows}")
        return plan
    
    def optimize_query(self, query: str) -> str:
        """
        Rewrite query for better performance
        
        Args:
            query: Original Cypher query
            
        Returns:
            Optimized query string
        """
        if not self.enable_query_rewriting:
            return query
        
        optimized = query
        
        # Optimization 1: Add LIMIT if missing and no aggregation
        if (
            self.optimization_level == OptimizationLevel.AGGRESSIVE and
            "LIMIT" not in optimized.upper() and
            "COUNT" not in optimized.upper() and
            "COLLECT" not in optimized.upper()
        ):
            optimized += " LIMIT 1000"
            logger.info("Added LIMIT 1000 to unbounded query")
        
        # Optimization 2: Move WHERE clause closer to MATCH
        # (This is a simplified example - real implementation would parse AST)
        if "WITH" in optimized.upper() and "WHERE" in optimized.upper():
            # Try to push WHERE clause earlier
            logger.debug("Consider moving WHERE clause earlier in query")
        
        # Optimization 3: Use WITH to limit intermediate results
        if (
            self.optimization_level == OptimizationLevel.AGGRESSIVE and
            optimized.count("MATCH") > 1 and
            "WITH" not in optimized.upper()
        ):
            logger.info("Consider using WITH to limit intermediate results")
        
        return optimized
    
    async def execute_cached(
        self,
        session: Any,
        query: str,
        params: Optional[Dict] = None,
        force_refresh: bool = False
    ) -> Any:
        """
        Execute query with caching
        
        Args:
            session: Neo4j session
            query: Cypher query
            params: Query parameters
            force_refresh: Force cache refresh
            
        Returns:
            Query result (from cache or fresh execution)
        """
        cache_key = self._generate_cache_key(query, params)
        query_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        
        # Check cache
        if self.enable_cache and not force_refresh:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                logger.debug(f"Cache HIT for query {query_hash}")
                self._query_stats[query_hash].cache_hits += 1
                return cached
            
            logger.debug(f"Cache MISS for query {query_hash}")
            self._query_stats[query_hash].cache_misses += 1
        
        # Execute query
        start_time = time.time()
        
        # Optimize query if enabled
        optimized_query = self.optimize_query(query)
        
        # Execute through session
        result = await session.run(optimized_query, params or {})
        result_list = [record.data() async for record in result]
        
        execution_time = time.time() - start_time
        
        # Update statistics
        self._query_stats[query_hash].update(execution_time)
        
        # Cache result
        if self.enable_cache:
            self._add_to_cache(cache_key, result_list, query_hash)
        
        logger.info(
            f"Query {query_hash} executed in {execution_time:.3f}s, "
            f"returned {len(result_list)} rows"
        )
        
        return result_list
    
    def recommend_indexes(self, query: str) -> List[Dict[str, Any]]:
        """
        Recommend indexes for query optimization
        
        Args:
            query: Cypher query to analyze
            
        Returns:
            List of index recommendations
        """
        recommendations = []
        
        # Extract WHERE clause properties
        where_properties = self._extract_where_properties(query)
        
        for prop in where_properties:
            if prop not in self._known_indexes:
                recommendations.append({
                    "type": "node_property_index",
                    "property": prop,
                    "reason": "Used in WHERE clause",
                    "priority": "high",
                    "create_statement": f"CREATE INDEX FOR (n:Node) ON (n.{prop})"
                })
        
        # Check for relationship queries
        if "()-[" in query and ":TYPE" in query:
            recommendations.append({
                "type": "relationship_index",
                "reason": "Relationship type filtering detected",
                "priority": "medium",
            })
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get query optimizer statistics
        
        Returns:
            Dictionary with statistics
        """
        total_queries = sum(stat.execution_count for stat in self._query_stats.values())
        total_cache_hits = sum(stat.cache_hits for stat in self._query_stats.values())
        total_cache_misses = sum(stat.cache_misses for stat in self._query_stats.values())
        
        cache_hit_rate = 0.0
        if total_cache_hits + total_cache_misses > 0:
            cache_hit_rate = total_cache_hits / (total_cache_hits + total_cache_misses)
        
        return {
            "total_unique_queries": len(self._query_stats),
            "total_executions": total_queries,
            "cache_size": len(self._cache),
            "cache_hit_rate": cache_hit_rate,
            "total_cache_hits": total_cache_hits,
            "total_cache_misses": total_cache_misses,
            "avg_query_time": self._calculate_average_query_time(),
            "optimization_level": self.optimization_level.value,
        }
    
    def clear_cache(self, query_pattern: Optional[str] = None):
        """
        Clear query cache
        
        Args:
            query_pattern: Optional pattern to match queries to clear
                          If None, clears entire cache
        """
        if query_pattern is None:
            cleared_count = len(self._cache)
            self._cache.clear()
            self._cache_access_order.clear()
            logger.info(f"Cleared entire cache ({cleared_count} entries)")
        else:
            # Clear matching entries
            keys_to_remove = [
                key for key in self._cache.keys()
                if query_pattern in key
            ]
            for key in keys_to_remove:
                del self._cache[key]
                if key in self._cache_access_order:
                    self._cache_access_order.remove(key)
            
            logger.info(f"Cleared {len(keys_to_remove)} cache entries matching pattern")
    
    def update_known_indexes(self, indexes: List[str]):
        """
        Update list of known database indexes
        
        Args:
            indexes: List of indexed properties
        """
        self._known_indexes = set(indexes)
        logger.info(f"Updated known indexes: {len(indexes)} indexes")
    
    # Private helper methods
    
    def _generate_cache_key(self, query: str, params: Optional[Dict]) -> str:
        """Generate cache key from query and parameters"""
        # Normalize query (remove extra whitespace)
        normalized_query = " ".join(query.split())
        
        # Include sorted params in key
        if params:
            params_str = json.dumps(params, sort_keys=True)
        else:
            params_str = ""
        
        return f"{normalized_query}|{params_str}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Retrieve result from cache if available and valid"""
        if cache_key not in self._cache:
            return None
        
        cached = self._cache[cache_key]
        
        # Check expiration
        if cached.is_expired():
            del self._cache[cache_key]
            if cache_key in self._cache_access_order:
                self._cache_access_order.remove(cache_key)
            return None
        
        # Update access tracking
        cached.access_count += 1
        cached.last_access = datetime.now()
        
        # Update LRU order
        if cache_key in self._cache_access_order:
            self._cache_access_order.remove(cache_key)
        self._cache_access_order.append(cache_key)
        
        return cached.result
    
    def _add_to_cache(self, cache_key: str, result: Any, query_hash: str):
        """Add result to cache with eviction if needed"""
        # Check cache size limit
        if len(self._cache) >= self.max_cache_size:
            self._evict_from_cache()
        
        # Create cached result
        cached = CachedResult(
            result=result,
            query_hash=query_hash,
            timestamp=datetime.now(),
            ttl_seconds=self.cache_ttl,
        )
        
        self._cache[cache_key] = cached
        self._cache_access_order.append(cache_key)
    
    def _evict_from_cache(self):
        """Evict entries from cache based on strategy"""
        if self.cache_strategy == CacheStrategy.LRU:
            # Remove least recently used
            if self._cache_access_order:
                oldest_key = self._cache_access_order.pop(0)
                del self._cache[oldest_key]
                logger.debug(f"Evicted LRU cache entry")
        
        elif self.cache_strategy == CacheStrategy.TTL:
            # Remove expired entries
            expired_keys = [
                key for key, cached in self._cache.items()
                if cached.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
                if key in self._cache_access_order:
                    self._cache_access_order.remove(key)
            
            # If still over limit, remove oldest by timestamp
            if len(self._cache) >= self.max_cache_size:
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].timestamp
                )
                del self._cache[oldest_key]
                if oldest_key in self._cache_access_order:
                    self._cache_access_order.remove(oldest_key)
    
    def _extract_where_properties(self, query: str) -> List[str]:
        """Extract property names used in WHERE clauses"""
        properties = []
        
        # Simple regex-like extraction (for demo purposes)
        # Real implementation would parse Cypher AST
        if "WHERE" in query.upper():
            where_part = query.upper().split("WHERE")[1].split("RETURN")[0]
            
            # Look for patterns like "n.property"
            import re
            pattern = r'\w+\.(\w+)'
            matches = re.findall(pattern, where_part)
            properties.extend(matches)
        
        return list(set(properties))  # Unique properties
    
    def _calculate_average_query_time(self) -> float:
        """Calculate average query execution time across all queries"""
        if not self._query_stats:
            return 0.0
        
        total_time = sum(stat.total_execution_time for stat in self._query_stats.values())
        total_count = sum(stat.execution_count for stat in self._query_stats.values())
        
        return total_time / total_count if total_count > 0 else 0.0


class QueryOptimizerConfig:
    """Configuration presets for QueryOptimizer"""
    
    @staticmethod
    def development() -> Dict[str, Any]:
        """Development configuration - aggressive caching"""
        return {
            "enable_cache": True,
            "cache_ttl": 600,  # 10 minutes
            "max_cache_size": 500,
            "cache_strategy": CacheStrategy.TTL,
            "optimization_level": OptimizationLevel.BASIC,
            "enable_query_rewriting": True,
        }
    
    @staticmethod
    def production() -> Dict[str, Any]:
        """Production configuration - conservative caching"""
        return {
            "enable_cache": True,
            "cache_ttl": 300,  # 5 minutes
            "max_cache_size": 1000,
            "cache_strategy": CacheStrategy.ADAPTIVE,
            "optimization_level": OptimizationLevel.AGGRESSIVE,
            "enable_query_rewriting": True,
        }
    
    @staticmethod
    def no_cache() -> Dict[str, Any]:
        """No caching - for testing"""
        return {
            "enable_cache": False,
            "optimization_level": OptimizationLevel.BASIC,
            "enable_query_rewriting": False,
        }
