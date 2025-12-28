"""Check Article properties and fix fulltext index."""
import os
from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', os.getenv('NEO4J_PASSWORD', 'password')))
session = driver.session()

print("=== Article properties ===")
result = session.run('MATCH (a:Article) RETURN keys(a) as props LIMIT 1')
rec = result.single()
print('Article properties:', rec['props'])

print("\n=== Recreating fulltext index on correct field ===")
# Drop old index
try:
    session.run("DROP INDEX article_fulltext IF EXISTS")
    print("Dropped old index")
except Exception as e:
    print(f"Could not drop: {e}")

# Create new index on full_text field
session.run("""
CREATE FULLTEXT INDEX article_fulltext IF NOT EXISTS 
FOR (a:Article) ON EACH [a.title, a.full_text, a.id]
""")
print("Created new fulltext index on [title, full_text, id]")

print("\n=== Testing search again ===")
result = session.run('''
CALL db.index.fulltext.queryNodes('article_fulltext', 'Bảng 5') YIELD node, score
RETURN node.id, node.title, score
LIMIT 5
''')
for r in result:
    print(f"  {r['node.id']} (score={r['score']:.2f}): {r['node.title']}")

session.close()
driver.close()
print("\nDone!")
