"""Quick search in Neo4j for summer term content."""
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    auth=(os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'password'))
)

with driver.session() as session:
    # Check what properties Article nodes have
    print("=== Properties of Article nodes ===\n")
    result = session.run("""
        MATCH (a:Article)
        WHERE a.id = 'dieu_14'
        RETURN keys(a) as props, a.title as title, 
               substring(coalesce(a.full_text, 'NULL'), 0, 500) as full_text,
               substring(coalesce(a.content, 'NULL'), 0, 500) as content
    """)
    
    for record in result:
        print(f"Properties: {record['props']}")
        print(f"Title: {record['title']}")
        print(f"full_text: {record['full_text']}")
        print(f"content: {record['content']}")
        print("-" * 50)
    
    # Also check in full_text
    print("\n=== Searching for 'hè' in full_text ===\n")
    result2 = session.run("""
        MATCH (n)
        WHERE n.full_text CONTAINS 'học kỳ hè' OR n.full_text CONTAINS 'hè'
        RETURN n.id as id, labels(n)[0] as label, 
               substring(n.full_text, 0, 200) as text
        LIMIT 10
    """)
    
    count = 0
    for record in result2:
        count += 1
        print(f"[{record['label']}] {record['id']}")
        print(f"  Text: {record['text']}...")
    
    if count == 0:
        print("No content found with 'hè' in full_text")

driver.close()
