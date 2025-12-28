#!/usr/bin/env python3
"""Check if Neo4j has Articles about English"""
import os
from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', os.getenv('NEO4J_PASSWORD', 'password')))
session = driver.session(database='neo4j')

print("=== Articles containing 'tiếng anh' ===")
result = session.run("""
    MATCH (a:Article) 
    WHERE toLower(a.title) CONTAINS 'tiếng anh' 
       OR toLower(a.full_text) CONTAINS 'tiếng anh'
    RETURN a.title as title, substring(a.full_text, 0, 100) as content
    LIMIT 10
""")

articles = list(result)
print(f"Found {len(articles)} articles")
for r in articles:
    print(f"  - {r['title']}")

print("\n=== Articles containing 'ngoại ngữ' ===")
result = session.run("""
    MATCH (a:Article) 
    WHERE toLower(a.title) CONTAINS 'ngoại ngữ' 
       OR toLower(a.full_text) CONTAINS 'ngoại ngữ'
    RETURN a.title as title
    LIMIT 10
""")

articles = list(result)
print(f"Found {len(articles)} articles")
for r in articles:
    print(f"  - {r['title']}")

print("\n=== Sample Article titles ===")
result = session.run("MATCH (a:Article) RETURN a.title LIMIT 10")
for r in result:
    print(f"  - {r['a.title']}")

driver.close()
