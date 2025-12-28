"""
Test RAG với câu hỏi thực tế
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adapters.weaviate_vector_adapter import WeaviateVectorAdapter
from core.domain.models import SearchQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

async def test_query(query_text: str):
    print("\n" + "=" * 80)
    print(f"QUERY: {query_text}")
    print("=" * 80)
    
    # Initialize
    embed_model = HuggingFaceEmbedding(
        model_name="intfloat/multilingual-e5-base",
        cache_folder=r"C:\Users\admin\AppData\Local\llama_index"
    )
    
    adapter = WeaviateVectorAdapter(
        weaviate_url="http://localhost:8090",
        embedding_model=embed_model
    )
    
    # Search
    query = SearchQuery(text=query_text, top_k=3)
    results = await adapter.search(query)
    
    print(f"\nTìm thấy {len(results)} kết quả:\n")
    
    for i, result in enumerate(results, 1):
        print(f"--- KẾT QUẢ {i} ---")
        print(f"Điều/Khoản: {result.metadata.extra.get('article', 'N/A')}")
        print(f"Loại: {result.metadata.extra.get('structure_type', 'N/A')}")
        print(f"Nội dung:\n{result.text}\n")
    
    adapter.close()

if __name__ == "__main__":
    # Test với câu hỏi về học phí
    asyncio.run(test_query("Học phí học kỳ hè tính như thế nào?"))
