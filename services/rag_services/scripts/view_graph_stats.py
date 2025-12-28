#!/usr/bin/env python3
"""View Neo4j graph statistics after LLM extraction."""
import os
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

print("\n" + "="*80)
print("📊 NEO4J KNOWLEDGE GRAPH STATISTICS")
print("="*80)

with driver.session() as session:
    # Count nodes by type
    print("\n🔢 Node Counts:")
    result = session.run("MATCH (n:Article) RETURN count(n) as count")
    articles = result.single()["count"]
    print(f"   Articles: {articles}")
    
    result = session.run("MATCH (n:Entity) RETURN count(n) as count")
    entities = result.single()["count"]
    print(f"   Entities: {entities}")
    
    result = session.run("MATCH (n:Rule) RETURN count(n) as count")
    rules = result.single()["count"]
    print(f"   Rules: {rules}")
    
    # Count relationships
    print("\n🔗 Relationship Counts:")
    result = session.run("MATCH ()-[r:MENTIONS]->() RETURN count(r) as count")
    mentions = result.single()["count"]
    print(f"   MENTIONS: {mentions}")
    
    result = session.run("MATCH ()-[r:DEFINES_RULE]->() RETURN count(r) as count")
    defines = result.single()["count"]
    print(f"   DEFINES_RULE: {defines}")
    
    # Sample entities
    print("\n👥 Sample Entities (first 10):")
    result = session.run("""
        MATCH (e:Entity)
        RETURN e.name as name, e.type as type, e.description as description
        LIMIT 10
    """)
    for i, record in enumerate(result, 1):
        print(f"   {i}. {record['name']} ({record['type']})")
        if record['description']:
            desc = record['description'][:80] + "..." if len(record['description']) > 80 else record['description']
            print(f"      {desc}")
    
    # Sample rules
    print("\n📋 Sample Rules (first 10):")
    result = session.run("""
        MATCH (r:Rule)
        RETURN r.id as id, r.type as type, r.description as description
        LIMIT 10
    """)
    for i, record in enumerate(result, 1):
        print(f"   {i}. {record['id']} ({record['type']})")
        if record['description']:
            desc = record['description'][:80] + "..." if len(record['description']) > 80 else record['description']
            print(f"      {desc}")
    
    # Entity types distribution
    print("\n📊 Entity Types Distribution:")
    result = session.run("""
        MATCH (e:Entity)
        RETURN e.type as type, count(*) as count
        ORDER BY count DESC
    """)
    for record in result:
        print(f"   {record['type']}: {record['count']}")
    
    # Rule types distribution
    print("\n📊 Rule Types Distribution:")
    result = session.run("""
        MATCH (r:Rule)
        RETURN r.type as type, count(*) as count
        ORDER BY count DESC
    """)
    for record in result:
        print(f"   {record['type']}: {record['count']}")

driver.close()
print("\n" + "="*80)
print("✅ Done!")
print("="*80 + "\n")
