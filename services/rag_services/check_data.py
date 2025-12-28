#!/usr/bin/env python3
"""Check data in Neo4j and test search"""
import os
from neo4j import GraphDatabase

d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', os.getenv('NEO4J_PASSWORD', 'password')))
s = d.session()

print("=== Articles liên quan đến tiếng Anh/ngoại ngữ ===")
r = list(s.run("""
    MATCH (a:Article) 
    WHERE toLower(a.title) CONTAINS 'tiếng anh' 
       OR toLower(a.title) CONTAINS 'ngoại ngữ'
       OR toLower(a.title) CONTAINS 'english'
    RETURN a.id, a.title LIMIT 10
"""))
for x in r:
    print(f"  {x[0]}: {x[1][:80]}")

print(f"\nTìm thấy: {len(r)} articles")

print("\n=== Entities liên quan đến tiếng Anh ===")
r2 = list(s.run("""
    MATCH (e:Entity)
    WHERE toLower(e.text) CONTAINS 'tiếng anh'
       OR toLower(e.text) CONTAINS 'ngoại ngữ'
       OR toLower(e.normalized_text) CONTAINS 'tiếng anh'
    RETURN e.text, e.type, e.normalized_text LIMIT 10
"""))
for x in r2:
    print(f"  {x[0]} ({x[1]})")

print(f"\nTìm thấy: {len(r2)} entities")

d.close()
