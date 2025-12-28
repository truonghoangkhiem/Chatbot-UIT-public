#!/usr/bin/env python3
"""Debug RAG Adapter directly to see what orchestrator receives"""
import asyncio
import sys
sys.path.insert(0, '.')

from app.adapters.rag_adapter import RAGServiceAdapter

async def test_rag():
    adapter = RAGServiceAdapter(
        rag_service_url="http://localhost:8000",
        timeout=60
    )
    
    query = "quy định về tiếng anh của trường thế nào"
    print(f"Query: {query}")
    print("=" * 60)
    
    try:
        result = await adapter.retrieve_context(
            query=query,
            top_k=5,
            search_mode="hybrid",
            use_rerank=True
        )
        
        print(f"\nResults from RAG Adapter:")
        print(f"Query: {result.get('query')}")
        docs = result.get('retrieved_documents', [])
        print(f"Documents: {len(docs)}")
        
        for i, doc in enumerate(docs):
            title = doc.get('title', 'N/A')
            text = doc.get('text', '')[:150]
            score = doc.get('score', 0)
            print(f"\n{i+1}. {title}")
            print(f"   Score: {score}")
            print(f"   Text: {text}...")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await adapter.close()

if __name__ == "__main__":
    asyncio.run(test_rag())
