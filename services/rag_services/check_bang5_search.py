"""Check if Bảng 5 is searchable in Neo4j."""
import os
from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', os.getenv('NEO4J_PASSWORD', 'password')))
session = driver.session()

print("=== Checking fulltext indexes ===")
result = session.run("SHOW INDEXES")
for r in result:
    if 'fulltext' in str(r.get('type', '')).lower():
        print(f"Index: {r['name']} on {r['labelsOrTypes']} ({r['properties']})")

print("\n=== Searching for 'Bảng 5' in articles ===")
try:
    result = session.run('''
    CALL db.index.fulltext.queryNodes('article_fulltext', 'Bảng 5') YIELD node, score
    RETURN node.id, node.title, score, substring(node.full_text, 0, 200) as excerpt
    LIMIT 5
    ''')
    found = False
    for r in result:
        found = True
        print(f"{r['node.id']} (score={r['score']:.2f})")
        print(f"  Title: {r['node.title']}")
        print(f"  Has 'Bảng 5': {'Bảng 5' in (r['excerpt'] or '')}")
    if not found:
        print("No results found!")
except Exception as e:
    print(f"Error: {e}")

print("\n=== Direct check: Articles containing 'Bảng 5' ===")
result = session.run('''
MATCH (a:Article) 
WHERE a.full_text CONTAINS 'Bảng 5' OR a.full_text CONTAINS 'bang_5'
RETURN a.id, a.title
''')
for r in result:
    print(f"  {r['a.id']}: {r['a.title']}")

session.close()
driver.close()
