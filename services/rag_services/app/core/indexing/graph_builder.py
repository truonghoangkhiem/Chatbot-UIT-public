"""
Neo4j Graph Builder for CatRAG Legal Document Pipeline.

This module ingests the extracted JSON data from the Two-Stage VLM+LLM pipeline
into Neo4j, following CatRAG architecture principles:

1. Entity Resolution: Merge entities by (normalized_text, type) to avoid duplicates
2. Category Routing: Create category nodes for topic-based search
3. Structure Ingestion: Build Document -> Article -> Clause hierarchy
4. Semantic Relations: Create edges between merged entities

Author: Legal Document Processing Team
Date: 2024
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class GraphStats:
    """Statistics from graph building process."""
    documents: int = 0
    chapters: int = 0
    articles: int = 0
    clauses: int = 0
    entities: int = 0
    entities_merged: int = 0  # Entities that were merged (not newly created)
    structural_relations: int = 0
    semantic_relations: int = 0
    mentions: int = 0
    categories: int = 0
    modifications: int = 0  # Legal modification relationships created


class Neo4jGraphBuilder:
    """
    Build Knowledge Graph in Neo4j from extracted legal document data.
    
    Implements CatRAG architecture:
    - Entity Resolution via MERGE on (normalized_text, type)
    - Category Routing via [:BELONGS_TO] relations
    - Hierarchical structure: Document -> Chapter -> Article -> Clause
    - Semantic relations between entities
    
    Example:
        ```python
        builder = Neo4jGraphBuilder(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
        
        with open("full_extraction.json") as f:
            data = json.load(f)
        
        stats = builder.build_graph(data, category="Quy chế Đào tạo")
        print(f"Created {stats.entities} entities, merged {stats.entities_merged}")
        
        builder.close()
        ```
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j bolt URI
            user: Neo4j username
            password: Neo4j password
            database: Database name
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("neo4j package required. Install: pip install neo4j")
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        
        # Verify connection
        try:
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # =========================================================================
    # CONSTRAINTS & INDEXES
    # =========================================================================
    
    def create_constraints(self):
        """
        Create constraints and indexes for optimal performance.
        
        Indexes on:
        - Document.id, Article.id, Clause.id (unique)
        - Entity.normalized_text + Entity.type (composite for merge)
        - Category.name (unique)
        """
        constraints = [
            # Unique constraints (also create indexes)
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (c:Chapter) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT clause_id IF NOT EXISTS FOR (c:Clause) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
            
            # Composite index for Entity Resolution (CRITICAL for CatRAG)
            # Entities are merged by (normalized_text, type) pair
            "CREATE INDEX entity_merge IF NOT EXISTS FOR (e:Entity) ON (e.normalized_text, e.type)",
            
            # Additional indexes for search
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE FULLTEXT INDEX entity_text IF NOT EXISTS FOR (e:Entity) ON EACH [e.text, e.normalized_text]",
        ]
        
        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.debug(f"Created: {constraint[:60]}...")
                except Exception as e:
                    # Constraint might already exist
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Constraint warning: {e}")
        
        logger.info("Constraints and indexes created/verified")
    
    def clear_database(self):
        """
        Clear all nodes and relationships. USE WITH CAUTION!
        """
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Database cleared - all nodes and relationships deleted")
    
    # =========================================================================
    # MAIN BUILD METHOD
    # =========================================================================
    
    def build_graph(
        self,
        extraction_data: Dict[str, Any],
        category: str = "Quy chế Đào tạo",
        clear_first: bool = False
    ) -> GraphStats:
        """
        Build complete knowledge graph from extraction data.
        
        Args:
            extraction_data: JSON data from Two-Stage pipeline
            category: Category name for routing (CatRAG)
            clear_first: Whether to clear database before building
            
        Returns:
            GraphStats with counts of created nodes/relations
        """
        stats = GraphStats()
        
        if clear_first:
            self.clear_database()
        
        # Ensure constraints exist
        self.create_constraints()
        
        # Extract stage data
        stage1 = extraction_data.get("stage1_structure", {})
        stage2 = extraction_data.get("stage2_semantic", {})
        
        with self.driver.session(database=self.database) as session:
            # 1. Create Category node (CatRAG routing)
            stats.categories = session.execute_write(
                self._create_category, category
            )
            
            # 2. Create structural hierarchy
            doc_stats = session.execute_write(
                self._create_document_structure, stage1, category
            )
            stats.documents = doc_stats["documents"]
            stats.chapters = doc_stats["chapters"]
            stats.articles = doc_stats["articles"]
            stats.clauses = doc_stats["clauses"]
            stats.structural_relations = doc_stats["relations"]
            
            # 3. Create/Merge entities with Entity Resolution
            entity_stats = session.execute_write(
                self._create_entities_with_resolution, stage2
            )
            stats.entities = entity_stats["created"]
            stats.entities_merged = entity_stats["merged"]
            stats.mentions = entity_stats["mentions"]
            
            # 4. Create semantic relations between entities
            stats.semantic_relations = session.execute_write(
                self._create_semantic_relations, stage2
            )
            
            # 5. Process legal modifications (amendments, replacements, etc.)
            modifications_list = stage2.get("modifications", [])
            if modifications_list:
                stats.modifications = session.execute_write(
                    self._process_modifications, modifications_list
                )
        
        logger.info(
            f"Graph built: {stats.documents} docs, {stats.articles} articles, "
            f"{stats.clauses} clauses, {stats.entities} entities "
            f"({stats.entities_merged} merged), {stats.semantic_relations} semantic relations, "
            f"{stats.modifications} modifications"
        )
        
        return stats
    
    # =========================================================================
    # CATEGORY ROUTING (CatRAG)
    # =========================================================================
    
    @staticmethod
    def _create_category(tx, category_name: str) -> int:
        """
        Create Category node for topic-based routing.
        
        CatRAG uses categories to route queries to relevant subgraphs.
        All Articles and Clauses will have [:BELONGS_TO] relation to this category.
        """
        query = """
        MERGE (c:Category {name: $name})
        ON CREATE SET 
            c.created_at = datetime(),
            c.description = 'Legal document category for routing'
        RETURN c
        """
        result = tx.run(query, name=category_name)
        count = len(list(result))
        logger.info(f"Category '{category_name}' created/verified")
        return count
    
    # =========================================================================
    # STRUCTURAL HIERARCHY
    # =========================================================================
    
    @staticmethod
    def _create_document_structure(
        tx, 
        stage1_data: Dict[str, Any],
        category: str
    ) -> Dict[str, int]:
        """
        Create document structure hierarchy.
        
        Structure: Document -[:CONTAINS]-> Chapter -[:CONTAINS]-> Article -[:CONTAINS]-> Clause
        
        Also creates:
        - [:BELONGS_TO] from Article/Clause to Category (CatRAG routing)
        - [:FOLLOWS] between sequential articles
        """
        stats = {"documents": 0, "chapters": 0, "articles": 0, "clauses": 0, "relations": 0}
        
        # --- Create Document node ---
        doc_data = stage1_data.get("document")
        if doc_data:
            query = """
            MERGE (d:Document {id: $id})
            SET d.title = $title,
                d.full_text = $full_text,
                d.page_range = $page_range,
                d.source = $source,
                d.updated_at = datetime()
            RETURN d
            """
            tx.run(query,
                id=doc_data["id"],
                title=doc_data.get("title", ""),
                full_text=doc_data.get("full_text", ""),
                page_range=doc_data.get("page_range", []),
                source=stage1_data.get("source", "")
            )
            stats["documents"] = 1
            logger.debug(f"Created Document: {doc_data['id']}")
        
        # --- Create Chapter nodes ---
        for chapter in stage1_data.get("chapters", []):
            query = """
            MERGE (ch:Chapter {id: $id})
            SET ch.title = $title,
                ch.full_text = $full_text,
                ch.page_range = $page_range,
                ch.updated_at = datetime()
            
            WITH ch
            MATCH (cat:Category {name: $category})
            MERGE (ch)-[:BELONGS_TO]->(cat)
            
            RETURN ch
            """
            tx.run(query,
                id=chapter["id"],
                title=chapter.get("title", ""),
                full_text=chapter.get("full_text", ""),
                page_range=chapter.get("page_range", []),
                category=category
            )
            stats["chapters"] += 1
        
        # --- Create Article nodes ---
        for article in stage1_data.get("articles", []):
            query = """
            MERGE (a:Article {id: $id})
            SET a.title = $title,
                a.full_text = $full_text,
                a.page_range = $page_range,
                a.updated_at = datetime()
            
            WITH a
            MATCH (cat:Category {name: $category})
            MERGE (a)-[:BELONGS_TO]->(cat)
            
            RETURN a
            """
            tx.run(query,
                id=article["id"],
                title=article.get("title", ""),
                full_text=article.get("full_text", ""),
                page_range=article.get("page_range", []),
                category=category
            )
            stats["articles"] += 1
            logger.debug(f"Created Article: {article['id']}")
        
        # --- Create Clause nodes ---
        for clause in stage1_data.get("clauses", []):
            query = """
            MERGE (cl:Clause {id: $id})
            SET cl.title = $title,
                cl.full_text = $full_text,
                cl.page_range = $page_range,
                cl.updated_at = datetime()
            
            WITH cl
            MATCH (cat:Category {name: $category})
            MERGE (cl)-[:BELONGS_TO]->(cat)
            
            RETURN cl
            """
            tx.run(query,
                id=clause["id"],
                title=clause.get("title", ""),
                full_text=clause.get("full_text", ""),
                page_range=clause.get("page_range", []),
                category=category
            )
            stats["clauses"] += 1
            logger.debug(f"Created Clause: {clause['id']}")
        
        # --- Create structural relations ---
        for rel in stage1_data.get("relations", []):
            source_id = rel["source"]
            target_id = rel["target"]
            rel_type = rel["type"]
            
            # Dynamic query based on relation type
            query = f"""
            MATCH (source) WHERE source.id = $source_id
            MATCH (target) WHERE target.id = $target_id
            MERGE (source)-[r:{rel_type}]->(target)
            RETURN r
            """
            result = tx.run(query, source_id=source_id, target_id=target_id)
            if list(result):
                stats["relations"] += 1
        
        logger.info(
            f"Structure created: {stats['documents']} docs, {stats['chapters']} chapters, "
            f"{stats['articles']} articles, {stats['clauses']} clauses, {stats['relations']} relations"
        )
        
        return stats
    
    # =========================================================================
    # ENTITY RESOLUTION (CRITICAL - CatRAG Core)
    # =========================================================================
    
    @staticmethod
    def _create_entities_with_resolution(
        tx,
        stage2_data: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Create Entity nodes with resolution/merging.
        
        CRITICAL CatRAG LOGIC:
        - Entities are MERGED by (normalized_text, type) pair
        - This ensures "sinh viên" mentioned in multiple articles 
          points to the SAME node
        - Creates [:MENTIONS] relation from Article/Clause to Entity
        
        Example:
            Article "dieu_1" mentions "sinh viên"
            Article "dieu_2" also mentions "sinh viên"
            
            Result: ONE Entity node, TWO [:MENTIONS] relations
        """
        stats = {"created": 0, "merged": 0, "mentions": 0}
        
        # Track which entities we've seen to count merges
        seen_entities = set()
        
        for entity in stage2_data.get("entities", []):
            entity_id = entity["id"]  # Local ID like "dieu_1_ent_1"
            entity_type = entity["type"]
            entity_text = entity.get("text", "")
            normalized_text = entity.get("normalized", entity_text)
            source_article_id = entity.get("source_article_id", "")
            confidence = entity.get("confidence", 1.0)
            
            # Key for merge: (normalized_text, type)
            merge_key = (normalized_text.lower().strip(), entity_type)
            
            # MERGE Entity by (normalized_text, type) - NOT by local ID!
            # This is the core of Entity Resolution
            query = """
            MERGE (e:Entity {normalized_text: $normalized_text, type: $type})
            ON CREATE SET 
                e.text = $text,
                e.created_at = datetime(),
                e.mention_count = 1
            ON MATCH SET 
                e.mention_count = coalesce(e.mention_count, 0) + 1,
                e.updated_at = datetime()
            
            WITH e
            
            // Create MENTIONS relation from source Article/Clause
            OPTIONAL MATCH (source) 
            WHERE source.id = $source_id 
              AND (source:Article OR source:Clause)
            
            FOREACH (_ IN CASE WHEN source IS NOT NULL THEN [1] ELSE [] END |
                MERGE (source)-[m:MENTIONS]->(e)
                ON CREATE SET 
                    m.confidence = $confidence,
                    m.local_id = $local_id,
                    m.created_at = datetime()
            )
            
            RETURN e.mention_count AS count
            """
            
            result = tx.run(query,
                normalized_text=normalized_text.lower().strip(),
                type=entity_type,
                text=entity_text,
                source_id=source_article_id,
                confidence=confidence,
                local_id=entity_id
            )
            
            record = result.single()
            mention_count = record["count"] if record else 1
            
            if merge_key in seen_entities:
                stats["merged"] += 1
            else:
                stats["created"] += 1
                seen_entities.add(merge_key)
            
            stats["mentions"] += 1
        
        logger.info(
            f"Entities: {stats['created']} created, {stats['merged']} merged, "
            f"{stats['mentions']} mentions"
        )
        
        return stats
    
    # =========================================================================
    # SEMANTIC RELATIONS
    # =========================================================================
    
    @staticmethod
    def _create_semantic_relations(
        tx,
        stage2_data: Dict[str, Any]
    ) -> int:
        """
        Create semantic relations between entities.
        
        Relations like YEU_CAU, AP_DUNG_CHO, DIEU_KIEN_TIEN_QUYET
        connect entities extracted from the same article.
        
        Important: We need to find the MERGED entities (by normalized_text + type),
        not the local IDs.
        """
        count = 0
        entities = stage2_data.get("entities", [])
        
        # Build lookup: local_id -> (normalized_text, type)
        entity_lookup = {}
        for ent in entities:
            local_id = ent["id"]
            entity_lookup[local_id] = {
                "normalized_text": ent.get("normalized", ent.get("text", "")).lower().strip(),
                "type": ent["type"]
            }
        
        for rel in stage2_data.get("relations", []):
            source_id = rel["source_id"]  # Local ID like "dieu_1_ent_1"
            target_id = rel["target_id"]
            rel_type = rel["type"]
            confidence = rel.get("confidence", 1.0)
            evidence = rel.get("evidence", "")
            source_article = rel.get("source_article_id", "")
            
            # Lookup the normalized entity info
            source_info = entity_lookup.get(source_id)
            target_info = entity_lookup.get(target_id)
            
            if not source_info or not target_info:
                logger.warning(f"Relation skipped: entity not found ({source_id} or {target_id})")
                continue
            
            # Find merged entities and create relation
            query = f"""
            MATCH (source:Entity {{
                normalized_text: $source_text, 
                type: $source_type
            }})
            MATCH (target:Entity {{
                normalized_text: $target_text, 
                type: $target_type
            }})
            MERGE (source)-[r:{rel_type}]->(target)
            ON CREATE SET 
                r.confidence = $confidence,
                r.evidence = $evidence,
                r.source_article = $source_article,
                r.created_at = datetime()
            RETURN r
            """
            
            result = tx.run(query,
                source_text=source_info["normalized_text"],
                source_type=source_info["type"],
                target_text=target_info["normalized_text"],
                target_type=target_info["type"],
                confidence=confidence,
                evidence=evidence,
                source_article=source_article
            )
            
            if list(result):
                count += 1
        
        logger.info(f"Semantic relations created: {count}")
        return count
    
    # =========================================================================
    # LEGAL MODIFICATIONS PROCESSING
    # =========================================================================
    
    @staticmethod
    def _process_modifications(
        tx,
        modifications_list: List[Dict[str, Any]]
    ) -> int:
        """
        Process legal modifications (amendments, replacements, supplements, repeals).
        
        For each modification:
        1. Find the Source Node (new regulation) by source_text_id
        2. Find the Target Node (old regulation) by document signature, article, clause
        3. Create modification relationship (AMENDS, REPLACES, SUPPLEMENTS, REPEALS)
        4. Update status of Target Node based on action type
        
        Args:
            tx: Neo4j transaction
            modifications_list: List of modification dictionaries
            
        Returns:
            Number of modification relationships created
        """
        count = 0
        
        # Mapping of action to target status
        action_to_status = {
            "AMENDS": "amended",
            "REPLACES": "expired",
            "SUPPLEMENTS": "amended",
            "REPEALS": "expired"
        }
        
        for mod in modifications_list:
            action = mod.get("action", "AMENDS")
            source_text_id = mod.get("source_text_id", "")
            target_doc_sig = mod.get("target_document_signature", "")
            target_article = mod.get("target_article", "")
            target_clause = mod.get("target_clause", "")
            effective_date = mod.get("effective_date")
            description = mod.get("description", "")
            
            if not source_text_id or not target_doc_sig:
                logger.warning(f"Modification skipped: missing source_text_id or target_document_signature")
                continue
            
            # Determine target status based on action
            new_status = action_to_status.get(action, "amended")
            
            # Build dynamic Cypher query to find and link nodes
            # The query tries to match target by document signature and optionally article/clause
            query = f"""
            // Find source node (the new regulation making the modification)
            MATCH (source)
            WHERE source.id = $source_text_id
              AND (source:Article OR source:Clause OR source:Document)
            
            // Find target node by document signature pattern
            // Try to match against Document, Article, or Clause nodes
            OPTIONAL MATCH (target_doc:Document)
            WHERE target_doc.id CONTAINS $target_doc_sig
               OR target_doc.title CONTAINS $target_doc_sig
            
            // Try to find target article within that document or by ID pattern
            OPTIONAL MATCH (target_art:Article)
            WHERE (target_art.id CONTAINS $target_doc_sig OR 
                   target_art.title CONTAINS $target_article_pattern)
              AND ($target_article = '' OR 
                   target_art.title CONTAINS $target_article OR 
                   target_art.id CONTAINS $target_article_id_pattern)
            
            // Try to find target clause
            OPTIONAL MATCH (target_cl:Clause)
            WHERE target_cl.id CONTAINS $target_doc_sig
              AND ($target_clause = '' OR 
                   target_cl.title CONTAINS $target_clause OR
                   target_cl.id CONTAINS $target_clause_id_pattern)
            
            // Determine the most specific target node
            WITH source,
                 CASE 
                     WHEN target_cl IS NOT NULL AND $target_clause <> '' THEN target_cl
                     WHEN target_art IS NOT NULL THEN target_art
                     WHEN target_doc IS NOT NULL THEN target_doc
                     ELSE NULL
                 END AS target
            
            WHERE target IS NOT NULL
            
            // Create the modification relationship
            MERGE (source)-[r:{action}]->(target)
            ON CREATE SET
                r.description = $description,
                r.effective_date = $effective_date,
                r.created_at = datetime()
            
            // Update target node status
            SET target.status = $new_status,
                target.modified_by = source.id,
                target.modification_date = CASE 
                    WHEN $effective_date IS NOT NULL THEN $effective_date 
                    ELSE toString(datetime()) 
                END
            
            RETURN source.id AS source_id, target.id AS target_id, type(r) AS rel_type
            """
            
            # Extract article/clause ID patterns (e.g., "Điều 4" -> "dieu_4")
            target_article_id = ""
            if target_article:
                # Convert "Điều 4" or "Điều 10" to "dieu_4" or "dieu_10"
                import re
                match = re.search(r'[Đđ]i[eề]u\s*(\d+)', target_article)
                if match:
                    target_article_id = f"dieu_{match.group(1)}"
            
            target_clause_id = ""
            if target_clause:
                # Convert "Khoản 3" to "khoan_3"
                import re
                match = re.search(r'[Kk]ho[aả]n\s*(\d+)', target_clause)
                if match:
                    target_clause_id = f"khoan_{match.group(1)}"
            
            try:
                result = tx.run(query,
                    source_text_id=source_text_id,
                    target_doc_sig=target_doc_sig,
                    target_article=target_article or "",
                    target_article_pattern=target_article or "",
                    target_article_id_pattern=target_article_id,
                    target_clause=target_clause or "",
                    target_clause_id_pattern=target_clause_id,
                    description=description,
                    effective_date=effective_date,
                    new_status=new_status
                )
                
                records = list(result)
                if records:
                    for rec in records:
                        logger.info(
                            f"Modification created: ({rec['source_id']})-[:{rec['rel_type']}]->({rec['target_id']})"
                        )
                        count += 1
                else:
                    logger.warning(
                        f"Modification target not found: {target_doc_sig} / {target_article} / {target_clause}"
                    )
                    
            except Exception as e:
                logger.error(f"Error creating modification relationship: {e}")
                continue
        
        logger.info(f"Legal modifications processed: {count}")
        return count
    
    # =========================================================================
    # QUERY HELPERS
    # =========================================================================
    
    def get_entity_by_text(self, text: str, entity_type: str = None) -> List[Dict]:
        """
        Find entity by text (for testing/debugging).
        """
        with self.driver.session(database=self.database) as session:
            if entity_type:
                query = """
                MATCH (e:Entity {type: $type})
                WHERE toLower(e.text) CONTAINS toLower($text)
                   OR toLower(e.normalized_text) CONTAINS toLower($text)
                RETURN e
                """
                result = session.run(query, text=text, type=entity_type)
            else:
                query = """
                MATCH (e:Entity)
                WHERE toLower(e.text) CONTAINS toLower($text)
                   OR toLower(e.normalized_text) CONTAINS toLower($text)
                RETURN e
                """
                result = session.run(query, text=text)
            
            return [dict(record["e"]) for record in result]
    
    def get_article_entities(self, article_id: str) -> List[Dict]:
        """
        Get all entities mentioned in an article.
        """
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (a {id: $article_id})-[:MENTIONS]->(e:Entity)
            RETURN e, a.title AS article_title
            """
            result = session.run(query, article_id=article_id)
            return [{"entity": dict(r["e"]), "article": r["article_title"]} for r in result]
    
    def get_category_articles(self, category_name: str) -> List[Dict]:
        """
        Get all articles in a category (CatRAG routing).
        """
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (a:Article)-[:BELONGS_TO]->(c:Category {name: $category})
            RETURN a.id AS id, a.title AS title
            ORDER BY a.id
            """
            result = session.run(query, category=category_name)
            return [dict(r) for r in result]
    
    def get_graph_stats(self) -> Dict[str, int]:
        """
        Get current graph statistics.
        """
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (d:Document) WITH count(d) AS docs
            MATCH (a:Article) WITH docs, count(a) AS articles
            MATCH (cl:Clause) WITH docs, articles, count(cl) AS clauses
            MATCH (e:Entity) WITH docs, articles, clauses, count(e) AS entities
            MATCH ()-[r:MENTIONS]->() WITH docs, articles, clauses, entities, count(r) AS mentions
            MATCH ()-[r]->() WITH docs, articles, clauses, entities, mentions, count(r) AS total_rels
            RETURN docs, articles, clauses, entities, mentions, total_rels
            """
            result = session.run(query)
            record = result.single()
            return dict(record) if record else {}


# =============================================================================
# CLI / Main
# =============================================================================

def main():
    """
    CLI to build graph from extraction JSON file.
    """
    import argparse
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Build Neo4j graph from extraction JSON")
    parser.add_argument("json_file", help="Path to extraction JSON file")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "password"))
    parser.add_argument("--category", default="Quy chế Đào tạo", help="Category name for routing")
    parser.add_argument("--clear", action="store_true", help="Clear database before building")
    
    args = parser.parse_args()
    
    # Load JSON
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"ERROR: File not found: {json_path}")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print("NEO4J GRAPH BUILDER")
    print(f"{'='*60}")
    print(f"Source: {json_path.name}")
    print(f"Neo4j: {args.uri}")
    print(f"Category: {args.category}")
    
    # Build graph
    try:
        with Neo4jGraphBuilder(
            uri=args.uri,
            user=args.user,
            password=args.password
        ) as builder:
            stats = builder.build_graph(
                extraction_data=data,
                category=args.category,
                clear_first=args.clear
            )
            
            print(f"\n{'='*60}")
            print("BUILD COMPLETE")
            print(f"{'='*60}")
            print(f"  Documents: {stats.documents}")
            print(f"  Articles:  {stats.articles}")
            print(f"  Clauses:   {stats.clauses}")
            print(f"  Entities:  {stats.entities} (merged: {stats.entities_merged})")
            print(f"  Mentions:  {stats.mentions}")
            print(f"  Structural Relations: {stats.structural_relations}")
            print(f"  Semantic Relations:   {stats.semantic_relations}")
            print(f"  Legal Modifications:  {stats.modifications}")
            
            # Show graph stats
            print(f"\nGraph Statistics:")
            graph_stats = builder.get_graph_stats()
            for key, value in graph_stats.items():
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
