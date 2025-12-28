#!/usr/bin/env python3
"""Test Neo4j search directly with new keyword logic"""
import asyncio
import os
import sys
sys.path.insert(0, 'c:/Users/admin/Downloads/Khiem/Chatbot-UIT/services/rag_services')

from adapters.graph.neo4j_adapter import Neo4jGraphAdapter

async def test_neo4j_search():
    adapter = Neo4jGraphAdapter(
        uri="bolt://localhost:7687",
        username="neo4j",
        password=os.getenv('NEO4J_PASSWORD', 'password'),
        database="neo4j"
    )
    
    # Test with compound term "tiếng anh"
    keywords = ["tiếng anh", "quy định"]
    
    print(f"Testing Neo4j search with keywords: {keywords}")
    print("=" * 60)
    
    try:
        results = await adapter.search_articles_by_keyword(keywords, limit=5)
        
        print(f"\nFound {len(results)} articles:")
        for i, r in enumerate(results):
            print(f"\n{i+1}. {r.get('title')}")
            print(f"   Score: {r.get('score', 'N/A')}")
            print(f"   Content: {r.get('content', '')[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_neo4j_search())
