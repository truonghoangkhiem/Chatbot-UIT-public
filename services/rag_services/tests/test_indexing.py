#!/usr/bin/env python3
"""
Test script để verify indexing đã thành công
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.weaviate_vector_adapter import WeaviateVectorAdapter
from infrastructure.embedding.huggingface_service import get_embedding_model
from core.domain.models import SearchQuery


async def test_indexing():
    """Test xem có retrieve được data đã index không"""
    
    print("=" * 80)
    print("TEST: Verify Indexing")
    print("=" * 80)
    
    # Initialize adapter
    print("\n⏳ Initializing Vector Adapter...")
    embedding_model = get_embedding_model()
    adapter = WeaviateVectorAdapter(
        weaviate_url="http://localhost:8080",
        embedding_model=embedding_model
    )
    print("✓ Adapter initialized")
    
    # Test queries
    test_queries = [
        "Điều kiện chuyển ngành là gì?",
        "Điều 19 quy định gì?",
        "Học phí tại UIT",
        "Tổ chức đánh giá môn học"
    ]
    
    for query_text in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query_text}")
        print(f"{'=' * 80}")
        
        query = SearchQuery(
            text=query_text,
            top_k=3,
            filters=None
        )
        
        results = await adapter.search(query)
        
        print(f"\n📊 Found {len(results)} results:")
        
        if results:
            for idx, result in enumerate(results, 1):
                print(f"\n  [{idx}] Score: {result.score:.4f}")
                print(f"      Chunk ID: {result.metadata.chunk_id}")
                print(f"      Title: {result.metadata.title}")
                print(f"      Doc Type: {result.metadata.doc_type}")
                
                # Show extra metadata
                if result.metadata.extra:
                    extra = result.metadata.extra
                    if 'source_id' in extra:
                        print(f"      Source ID: {extra['source_id']}")
                    if 'article_number' in extra:
                        print(f"      Article Number: {extra['article_number']}")
                    if 'structure_type' in extra:
                        print(f"      Type: {extra['structure_type']}")
                
                # Show content preview
                content = result.text[:200]
                print(f"      Content: {content}...")
        else:
            print("  ❌ No results found!")
    
    print(f"\n{'=' * 80}")
    print("TEST COMPLETED")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(test_indexing())
