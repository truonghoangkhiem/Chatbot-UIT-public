"""
Test RAG retrieval to verify vectors work correctly.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adapters.weaviate_vector_adapter import WeaviateVectorAdapter
from core.domain.models import SearchQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

async def main():
    print("\n" + "=" * 80)
    print("TESTING RAG VECTOR SEARCH")
    print("=" * 80)
    
    # Initialize embedding model
    print("\n1. Loading embedding model...")
    embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-base",
        cache_folder=r"C:\Users\admin\AppData\Local\llama_index"
    )
    print("   ✓ Model loaded")
    
    # Initialize Weaviate adapter
    print("\n2. Connecting to Weaviate...")
    adapter = WeaviateVectorAdapter(
        weaviate_url="http://localhost:8090",
        embedding_model=embed_model
    )
    print("   ✓ Connected")
    
    # Test queries
    test_queries = [
        "Học phí học kỳ hè tính như thế nào?",
        "Điều 1 quy định gì?",
        "Quyết định này có hiệu lực khi nào?",
    ]
    
    print("\n3. Testing vector search:")
    print("=" * 80)
    
    for query_text in test_queries:
        print(f"\nQuery: {query_text}")
        print("-" * 80)
        
        query = SearchQuery(text=query_text, top_k=2)
        results = await adapter.search(query)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n  Result {i}:")
                print(f"    Score: {result.score:.4f}")
                print(f"    Article: Điều {result.metadata.extra.get('article_number', 'N/A')}")
                print(f"    Text preview: {result.text[:200]}...")
        else:
            print("  ✗ No results found!")
    
    adapter.close()
    print("\n" + "=" * 80)
    print("✓ TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
