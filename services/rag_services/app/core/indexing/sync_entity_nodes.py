#!/usr/bin/env python3
"""
Sync Entity nodes from entities_json in Articles

This script ensures Entity nodes are properly created from the entities_json
property stored in Article nodes. This is important for:
1. Leiden community detection algorithm
2. Multi-hop graph traversal queries
3. Entity-based search

Usage:
    python scripts/sync_entity_nodes.py
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from neo4j import GraphDatabase


def setup_environment():
    """Load environment variables"""
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path, override=True)
    
    return {
        "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
        "neo4j_password": os.getenv("NEO4J_PASSWORD")
    }


def sync_entity_nodes(driver):
    """
    Sync Entity nodes from entities_json in Articles.
    
    This ensures:
    1. Each entity in entities_json becomes a separate :Entity node
    2. Each Article-Entity pair has a :MENTIONS relationship
    """
    print("\n" + "=" * 60)
    print("🔄 SYNC ENTITY NODES FROM entities_json")
    print("=" * 60)
    
    with driver.session() as session:
        # Step 1: Check current state
        print("\n1️⃣ Current state:")
        
        articles_with_json = session.run("""
            MATCH (a:Article)
            WHERE a.entities_json IS NOT NULL AND a.entities_json <> '' AND a.entities_json <> '[]'
            RETURN count(a) as count
        """).single()['count']
        print(f"   Articles with entities_json: {articles_with_json}")
        
        entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()['count']
        print(f"   Existing Entity nodes: {entity_count}")
        
        mentions_count = session.run("MATCH ()-[r:MENTIONS]->() RETURN count(r) as count").single()['count']
        print(f"   Existing MENTIONS relationships: {mentions_count}")
        
        # Step 2: Get all articles with entities_json
        print("\n2️⃣ Processing entities_json...")
        
        articles = session.run("""
            MATCH (a:Article)
            WHERE a.entities_json IS NOT NULL AND a.entities_json <> '' AND a.entities_json <> '[]'
            RETURN a.id as id, a.entities_json as entities_json
        """)
        
        total_entities_created = 0
        total_mentions_created = 0
        errors = 0
        
        for record in articles:
            article_id = record['id']
            entities_json = record['entities_json']
            
            try:
                # Parse JSON
                entities = json.loads(entities_json)
                
                for entity in entities:
                    name = entity.get('name', '').strip()
                    entity_type = entity.get('type', 'Unknown').strip()
                    
                    if not name:
                        continue
                    
                    # Create Entity node and MENTIONS relationship
                    result = session.run("""
                        MERGE (e:Entity {name: $name, type: $type})
                        WITH e
                        MATCH (a:Article {id: $article_id})
                        MERGE (a)-[r:MENTIONS]->(e)
                        RETURN 
                            CASE WHEN e.created IS NULL THEN true ELSE false END as new_entity,
                            CASE WHEN r.created IS NULL THEN true ELSE false END as new_mention
                    """, name=name, type=entity_type, article_id=article_id)
                    
                    # Note: This is a simplified count - actual new vs existing would need more logic
                    total_entities_created += 1
                    total_mentions_created += 1
                    
            except json.JSONDecodeError as e:
                print(f"   ⚠️ JSON error for {article_id}: {e}")
                errors += 1
            except Exception as e:
                print(f"   ⚠️ Error processing {article_id}: {e}")
                errors += 1
        
        print(f"   ✅ Processed entities: {total_entities_created}")
        print(f"   ✅ Created/verified MENTIONS: {total_mentions_created}")
        if errors:
            print(f"   ⚠️ Errors: {errors}")
        
        # Step 3: Verify final state
        print("\n3️⃣ Final state:")
        
        entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()['count']
        print(f"   Entity nodes: {entity_count}")
        
        mentions_count = session.run("MATCH ()-[r:MENTIONS]->() RETURN count(r) as count").single()['count']
        print(f"   MENTIONS relationships: {mentions_count}")
        
        # Check shared entities (important for Leiden)
        shared = session.run("""
            MATCH (a1:Article)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(a2:Article)
            WHERE a1.id < a2.id
            RETURN count(DISTINCT e) as shared
        """).single()['shared']
        print(f"   Shared entities (for Leiden): {shared}")
        
        # Top entities
        print("\n   📊 Top 5 entities:")
        top = session.run("""
            MATCH (a:Article)-[:MENTIONS]->(e:Entity)
            WITH e, count(a) as mentions
            ORDER BY mentions DESC
            LIMIT 5
            RETURN e.name, e.type, mentions
        """)
        for t in top:
            print(f"      - {t['e.name']} ({t['e.type']}): {t['mentions']} articles")


def main():
    config = setup_environment()
    
    driver = GraphDatabase.driver(
        config["neo4j_uri"],
        auth=(config["neo4j_user"], config["neo4j_password"])
    )
    
    try:
        sync_entity_nodes(driver)
        print("\n✅ Entity sync complete!")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
