"""Check Điều 9 content in Neo4j."""
import os
from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', os.getenv('NEO4J_PASSWORD', 'password')))
session = driver.session()

result = session.run('MATCH (a:Article {id: "dieu_9"}) RETURN a.full_text as txt')
rec = result.single()

if rec:
    txt = rec['txt'] or ""
    print(f"Length: {len(txt)}")
    print("=" * 50)
    if "Bảng 5" in txt or "bang_5" in txt:
        print("✅ Bảng 5 content IS MERGED into Điều 9")
    else:
        print("❌ Bảng 5 content NOT in Điều 9")
    
    if "Bảng 6" in txt or "bang_6" in txt:
        print("✅ Bảng 6 content IS MERGED into Điều 9")
    else:
        print("❌ Bảng 6 content NOT in Điều 9")
    print("=" * 50)
    print("\n--- FULL TEXT ---\n")
    print(txt)
else:
    print("Điều 9 not found in Neo4j")

session.close()
driver.close()
