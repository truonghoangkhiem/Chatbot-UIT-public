"""
Test RAG retrieval after indexing
"""
import requests
import json

def test_rag_search(query: str, top_k: int = 3):
    """Test RAG search endpoint"""
    url = "http://localhost:8000/v1/search"
    payload = {
        "query": query,
        "top_k": top_k,
        "use_reranking": False
    }
    
    print(f"\n{'='*80}")
    print(f"Testing query: {query}")
    print(f"{'='*80}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        print(f"\nStatus: {response.status_code}")
        print(f"Documents retrieved: {len(result.get('documents', []))}")
        
        for i, doc in enumerate(result.get('documents', []), 1):
            print(f"\n--- Document {i} ---")
            print(f"Content: {doc.get('content', '')[:200]}...")
            print(f"Score: {doc.get('score', 0):.4f}")
            metadata = doc.get('metadata', {})
            print(f"Metadata: article_number={metadata.get('article_number')}, "
                  f"doc_type={metadata.get('doc_type')}, "
                  f"source_id={metadata.get('source_id')}")
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    # Test queries related to the indexed content
    queries = [
        "Học phí học kỳ hè tính như thế nào?",
        "Điều kiện mở lớp học phần",
        "Sinh viên đăng ký tối đa bao nhiêu tín chỉ?",
        "Miễn học phần cần làm thủ tục gì?"
    ]
    
    for query in queries:
        test_rag_search(query)
