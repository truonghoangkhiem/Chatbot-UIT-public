#!/usr/bin/env python3
"""
Tạo các fulltext indexes cần thiết trong Neo4j
"""

from neo4j import GraphDatabase
import os

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

print("=" * 80)
print("CREATING NEO4J FULLTEXT INDEXES")
print("=" * 80)
print()

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def create_indexes(tx):
    """Create all required fulltext indexes"""
    
    indexes_to_create = [
        {
            "name": "mon_hoc_fulltext",
            "labels": ["MON_HOC"],
            "properties": ["ten_mon", "ma_mon", "mo_ta"]
        },
        {
            "name": "article_fulltext",
            "labels": ["Article"],
            "properties": ["title", "content", "article_id"]
        },
        {
            "name": "dieu_khoan_fulltext",
            "labels": ["DIEU_KHOAN"],
            "properties": ["title", "content"]
        }
    ]
    
    for idx_config in indexes_to_create:
        name = idx_config["name"]
        labels = idx_config["labels"]
        properties = idx_config["properties"]
        
        # Check if index exists
        check_query = f"""
        SHOW INDEXES
        YIELD name
        WHERE name = '{name}'
        RETURN count(*) as exists
        """
        
        result = tx.run(check_query)
        exists = result.single()["exists"] > 0
        
        if exists:
            print(f"✓ Index '{name}' already exists")
        else:
            # Create index using Cypher syntax for Neo4j 5.x
            labels_str = "|".join(labels)
            props_str = ", ".join([f"n.{p}" for p in properties])
            
            create_query = f"""
            CREATE FULLTEXT INDEX {name} IF NOT EXISTS
            FOR (n:{labels_str})
            ON EACH [{props_str}]
            """
            
            try:
                tx.run(create_query)
                print(f"✅ Created index '{name}' on {labels} with properties {properties}")
            except Exception as e:
                print(f"❌ Failed to create index '{name}': {e}")

try:
    with driver.session() as session:
        session.execute_write(create_indexes)
    
    print()
    print("=" * 80)
    print("✅ FULLTEXT INDEXES CREATION COMPLETED")
    print("=" * 80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    driver.close()
