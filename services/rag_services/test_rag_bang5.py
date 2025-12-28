"""Test RAG search for 'Bảng 5' directly."""
import asyncio
import aiohttp
import json

async def test_rag_search():
    async with aiohttp.ClientSession() as session:
        payload = {
            "query": "chứng chỉ bảng 5 là chứng chỉ gì",
            "top_k": 5,
            "search_mode": "hybrid",
            "use_rerank": True
        }
        
        print(f"Query: {payload['query']}")
        print("=" * 60)
        
        async with session.post(
            "http://localhost:8001/v1/search",
            json=payload
        ) as response:
            if response.status == 200:
                data = await response.json()
                hits = data.get("hits", [])
                print(f"Found {len(hits)} hits\n")
                
                for i, hit in enumerate(hits):
                    text = hit.get("text", "")[:500]
                    title = hit.get("title", "Unknown")
                    score = hit.get("score", 0)
                    
                    print(f"--- Hit {i+1} (score={score:.3f}) ---")
                    print(f"Title: {title}")
                    print(f"Has 'Bảng 5': {'Bảng 5' in text}")
                    print(f"Has table content: {'TOEIC' in text or 'IELTS' in text}")
                    print(f"Text preview: {text[:300]}...")
                    print()
            else:
                print(f"Error: {response.status}")
                print(await response.text())

asyncio.run(test_rag_search())
