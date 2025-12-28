from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

d = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"), 
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password"))
)
s = d.session()

# Tìm Điều 9
print("=== Tìm Điều 9 về chứng chỉ tốt nghiệp ===")
r = s.run("""
MATCH (n:Article) 
WHERE n.id CONTAINS 'dieu_9' OR n.title CONTAINS 'tốt nghiệp'
RETURN n.id, n.title, substring(n.full_text, 0, 800) as text
""").data()

for row in r:
    print(f"\nID: {row['n.id']}")
    print(f"Title: {row['n.title']}")
    print(f"Text: {row['text']}...")

# Tìm bảng 5
print("\n\n=== Tìm Bảng 5 - Chứng chỉ tiếng Anh ===")
r2 = s.run("""
MATCH (n) 
WHERE n.full_text CONTAINS 'Bảng 5' OR n.full_text CONTAINS 'TOEIC' OR n.full_text CONTAINS 'IELTS'
RETURN labels(n)[0] as type, n.id, n.title, substring(n.full_text, 0, 500) as text
LIMIT 5
""").data()

for row in r2:
    print(f"\nType: {row['type']}")
    print(f"ID: {row['n.id']}")
    print(f"Title: {row['n.title']}")
    print(f"Text: {row['text']}...")

d.close()
