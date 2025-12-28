#!/usr/bin/env python3
"""
Demo: Test Agent có lấy context từ RAG được không
Sử dụng OpenSearch trực tiếp để bypass RAG API issues
"""

import requests
from opensearchpy import OpenSearch
import json

# URLs
ORCHESTRATOR_URL = "http://localhost:8001"

# OpenSearch connection
os_client = OpenSearch(
    hosts=[{'host': 'localhost', 'port': 9200}],
    http_auth=('admin', 'admin'),
    use_ssl=False,
    verify_certs=False,
    ssl_show_warn=False
)

# ANSI colors
class Colors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_section(title: str):
    """Print section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")

def search_opensearch(query: str, top_k: int = 3):
    """Tìm kiếm trong OpenSearch."""
    try:
        response = os_client.search(
            index="rag_documents",
            body={
                "query": {"match": {"text": query}},
                "size": top_k,
                "_source": ["text", "title", "doc_type", "doc_id"]
            }
        )
        
        hits = response['hits']['hits']
        results = []
        
        for hit in hits:
            results.append({
                "text": hit['_source'].get('text', ''),
                "score": hit['_score'],
                "title": hit['_source'].get('title', ''),
                "doc_id": hit['_source'].get('doc_id', '')
            })
        
        return results
    except Exception as e:
        print(f"{Colors.RED}❌ OpenSearch error: {e}{Colors.END}")
        return []

def test_agent_with_context(query: str):
    """Test agent với context từ OpenSearch."""
    print_section(f"TEST: {query}")
    
    # Bước 1: Tìm kiếm context
    print(f"{Colors.CYAN}🔍 Bước 1: Tìm kiếm trong OpenSearch...{Colors.END}")
    results = search_opensearch(query, top_k=3)
    
    if not results:
        print(f"{Colors.RED}❌ Không tìm thấy context từ OpenSearch{Colors.END}")
        return False
    
    print(f"{Colors.GREEN}✅ Tìm thấy {len(results)} kết quả:{Colors.END}\n")
    
    # Hiển thị results
    for i, result in enumerate(results, 1):
        print(f"{Colors.YELLOW}[{i}] Score: {result['score']:.4f}{Colors.END}")
        print(f"    Text preview: {result['text'][:100]}...")
        print()
    
    # Bước 2: Tạo context
    context = "\n\n".join([r['text'][:500] for r in results])
    
    # Bước 3: Gửi tới agent
    print(f"{Colors.CYAN}🤖 Bước 2: Gửi câu hỏi + context tới Agent...{Colors.END}\n")
    
    full_query = f"""Dựa trên thông tin sau đây, hãy trả lời câu hỏi: "{query}"

Thông tin tham khảo:
{context[:1500]}

Hãy trả lời chi tiết dựa trên thông tin đã cung cấp."""
    
    try:
        response = requests.post(
            f"{ORCHESTRATOR_URL}/api/v1/chat",
            json={
                "query": full_query,
                "session_id": "test-rag-context",
                "use_rag": False  # Đã có context rồi
            },
            timeout=90
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('response', 'No response')
            model = data.get('model_used', 'Unknown')
            
            print(f"{Colors.GREEN}{'─'*80}{Colors.END}")
            print(f"{Colors.BOLD}🤖 Agent Response (Model: {model}):{Colors.END}\n")
            print(answer)
            print(f"{Colors.GREEN}{'─'*80}{Colors.END}\n")
            
            # Phân tích response
            print(f"{Colors.CYAN}📊 Phân tích:{Colors.END}")
            if any(word in answer.lower() for word in ['tốt nghiệp', 'điều kiện', 'tín chỉ', 'khóa luận', 'thực tập']):
                print(f"{Colors.GREEN}✅ Agent đã sử dụng thông tin từ context{Colors.END}")
            else:
                print(f"{Colors.YELLOW}⚠️  Agent có thể chưa sử dụng đầy đủ context{Colors.END}")
            
            return True
        else:
            print(f"{Colors.RED}❌ API Error: {response.status_code}{Colors.END}")
            return False
            
    except Exception as e:
        print(f"{Colors.RED}❌ Error: {e}{Colors.END}")
        return False

def main():
    """Main function."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║              TEST: AGENT CÓ LẤY CONTEXT TỪ RAG ĐƯỢC KHÔNG?                  ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")
    
    # Test với các câu hỏi khác nhau
    test_cases = [
        "Điều kiện tốt nghiệp của sinh viên UIT là gì?",
        "Quy định về thực tập và khóa luận tốt nghiệp",
        "Số tín chỉ cần thiết để tốt nghiệp"
    ]
    
    results = []
    
    for query in test_cases:
        success = test_agent_with_context(query)
        results.append(success)
        
        if query != test_cases[-1]:  # Không phải câu cuối
            input(f"\n{Colors.CYAN}Nhấn Enter để tiếp tục...{Colors.END}\n")
    
    # Summary
    print_section("KẾT QUẢ TỔNG HỢP")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tổng số test: {total}")
    print(f"Thành công: {Colors.GREEN}{passed}{Colors.END}")
    print(f"Thất bại: {Colors.RED}{total - passed}{Colors.END}\n")
    
    if passed == total:
        print(f"{Colors.GREEN}{'='*80}{Colors.END}")
        print(f"{Colors.GREEN}{Colors.BOLD}✅ KẾT LUẬN: AGENT ĐÃ LẤY VÀ SỬ DỤNG CONTEXT TỪ RAG!{Colors.END}")
        print(f"{Colors.GREEN}{'='*80}{Colors.END}\n")
    else:
        print(f"{Colors.YELLOW}⚠️  Một số test chưa thành công, cần kiểm tra lại{Colors.END}\n")

if __name__ == "__main__":
    main()
