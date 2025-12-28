#!/usr/bin/env python3
"""
Demo: Agent tương tác với RAG service
Test orchestrator agent gọi RAG để trả lời câu hỏi về chương trình đào tạo
"""

import asyncio
import requests
import json
from typing import Dict, Any

# URLs
ORCHESTRATOR_URL = "http://localhost:8001"  # Orchestrator service
RAG_URL = "http://localhost:8000"  # RAG service

# ANSI colors for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_section(title: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")

def print_user_query(query: str):
    """Print user query."""
    print(f"{Colors.BOLD}{Colors.BLUE}👤 Người dùng:{Colors.END} {query}")

def print_agent_response(response: str):
    """Print agent response."""
    print(f"{Colors.BOLD}{Colors.GREEN}🤖 Agent:{Colors.END} {response}")

def print_rag_results(results: list):
    """Print RAG search results."""
    print(f"\n{Colors.YELLOW}📚 RAG Context ({len(results)} kết quả):{Colors.END}")
    for i, result in enumerate(results, 1):
        print(f"\n  {Colors.BOLD}[{i}] {result.get('metadata', {}).get('title', 'N/A')}{Colors.END}")
        print(f"      Score: {result.get('score', 0):.4f}")
        print(f"      Preview: {result.get('text', '')[:150]}...")

def check_service(url: str, name: str) -> bool:
    """Check if a service is running."""
    try:
        # Try /docs endpoint since /health may not exist
        response = requests.get(f"{url}/docs", timeout=None)
        if response.status_code == 200:
            print(f"{Colors.GREEN}✓{Colors.END} {name} is running at {url}")
            return True
        else:
            print(f"{Colors.RED}✗{Colors.END} {name} returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"{Colors.RED}✗{Colors.END} {name} is not accessible at {url}")
        print(f"  Error: {e}")
        return False

def test_rag_search(query: str) -> Dict[str, Any]:
    """Test RAG search directly."""
    try:
        response = requests.post(
            f"{RAG_URL}/v1/search",  # RAG service uses /v1/search not /api/v1/search
            json={
                "query": query,
                "top_k": 3,
                "search_mode": "vector"
            },
            timeout=None  # No timeout
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"{Colors.RED}Error calling RAG:{Colors.END} {e}")
        return {"success": False, "error": str(e)}

def test_orchestrator_chat(message: str, session_id: str = "demo-session") -> Dict[str, Any]:
    """Test orchestrator chat."""
    try:
        response = requests.post(
            f"{ORCHESTRATOR_URL}/api/v1/chat",
            json={
                "query": message,  # API expects "query" not "message"
                "session_id": session_id,
                "use_rag": True,
                "rag_top_k": 3
            },
            timeout=None  # No timeout
        )
        response.raise_for_status()
        result = response.json()
        # Transform to expected format
        return {
            "success": True,
            "response": result.get("response", ""),
            "agent_used": result.get("model_used", "Unknown"),
            "sources": []  # Could extract from rag_context if needed
        }
    except Exception as e:
        print(f"{Colors.RED}Error calling Orchestrator:{Colors.END} {e}")
        return {"success": False, "error": str(e)}

def demo_rag_only():
    """Demo 1: Test RAG service trực tiếp."""
    print_section("DEMO 1: Test RAG Service Trực Tiếp")
    
    queries = [
        "Mục tiêu chung của chương trình giáo dục đại học của Trường",
        "Điều kiện tốt nghiệp ngành KHMT",
        "Các học phần bắt buộc"
    ]
    
    for query in queries:
        print_user_query(query)
        result = test_rag_search(query)
        
        if result.get("success"):
            results = result.get("results", [])
            print_rag_results(results)
        else:
            print(f"{Colors.RED}❌ Search failed: {result.get('error', 'Unknown error')}{Colors.END}")
        
        print()

def demo_orchestrator_with_rag():
    """Demo 2: Test Orchestrator agent sử dụng RAG."""
    print_section("DEMO 2: Agent Tương Tác Với RAG")
    
    # Các câu hỏi test
    questions = [
        {
            "query": "Mục tiêu chung của chương trình giáo dục đại học của Trường là gì?",
            "description": "Câu hỏi về thông tin chung - agent nên dùng RAG"
        },
        {
            "query": "Điều kiện tốt nghiệp của ngành Khoa học Máy tính là gì?",
            "description": "Câu hỏi cụ thể - agent nên dùng RAG"
        },
        {
            "query": "Hello, bạn là ai?",
            "description": "Câu hỏi chào hỏi - agent không cần RAG"
        },
        {
            "query": "Tóm tắt các học phần bắt buộc trong chương trình KHMT",
            "description": "Câu hỏi phức tạp - agent nên dùng RAG và tổng hợp"
        }
    ]
    
    session_id = "demo-rag-integration"
    
    for i, question in enumerate(questions, 1):
        print(f"\n{Colors.BOLD}Câu hỏi {i}/{len(questions)}:{Colors.END} {question['description']}")
        print_user_query(question["query"])
        
        # Gọi orchestrator
        result = test_orchestrator_chat(question["query"], session_id)
        
        if result.get("success"):
            response_text = result.get("response", "No response")
            agent_used = result.get("agent_used", "Unknown")
            
            print(f"\n{Colors.CYAN}Agent sử dụng:{Colors.END} {agent_used}")
            print_agent_response(response_text)
            
            # Nếu có context từ RAG
            if "context" in result or "sources" in result:
                print(f"\n{Colors.YELLOW}📑 Sources from RAG:{Colors.END}")
                sources = result.get("sources", [])
                if sources:
                    for j, source in enumerate(sources, 1):
                        print(f"  [{j}] {source}")
                else:
                    print("  (No sources returned)")
        else:
            print(f"{Colors.RED}❌ Chat failed: {result.get('error', 'Unknown error')}{Colors.END}")
        
        print(f"\n{Colors.CYAN}{'-'*80}{Colors.END}")

def interactive_demo():
    """Demo 3: Chế độ hỏi đáp tương tác."""
    print_section("DEMO 3: Chế Độ Hỏi Đáp Tương Tác")
    
    print(f"{Colors.YELLOW}Nhập câu hỏi (hoặc 'quit' để thoát):{Colors.END}\n")
    
    session_id = "interactive-demo"
    
    while True:
        try:
            query = input(f"{Colors.BOLD}{Colors.BLUE}👤 Bạn: {Colors.END}").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Colors.GREEN}Goodbye! 👋{Colors.END}\n")
                break
            
            # Test RAG trước
            print(f"\n{Colors.CYAN}🔍 Searching RAG...{Colors.END}")
            rag_result = test_rag_search(query)
            
            if rag_result.get("success"):
                results = rag_result.get("results", [])
                if results:
                    print(f"{Colors.GREEN}✓{Colors.END} Found {len(results)} relevant documents")
                else:
                    print(f"{Colors.YELLOW}⚠{Colors.END} No documents found")
            
            # Gọi orchestrator
            print(f"{Colors.CYAN}🤖 Agent processing...{Colors.END}\n")
            result = test_orchestrator_chat(query, session_id)
            
            if result.get("success"):
                response_text = result.get("response", "No response")
                agent_used = result.get("agent_used", "Unknown")
                
                print(f"{Colors.CYAN}Agent: {agent_used}{Colors.END}")
                print_agent_response(response_text)
            else:
                print(f"{Colors.RED}❌ Error: {result.get('error', 'Unknown error')}{Colors.END}")
            
            print()
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.GREEN}Goodbye! 👋{Colors.END}\n")
            break
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.END}\n")

def main():
    """Main demo function."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                    AGENT + RAG INTEGRATION DEMO                              ║")
    print("║          Test Orchestrator Agent với RAG Service                             ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")
    
    # Check services
    print_section("Kiểm Tra Services")
    
    rag_ok = check_service(RAG_URL, "RAG Service")
    orchestrator_ok = check_service(ORCHESTRATOR_URL, "Orchestrator Service")
    
    if not rag_ok:
        print(f"\n{Colors.YELLOW}⚠️  RAG Service chưa chạy!{Colors.END}")
        print(f"   Khởi động bằng: cd services/rag_services && python start_server.py")
    
    if not orchestrator_ok:
        print(f"\n{Colors.YELLOW}⚠️  Orchestrator Service chưa chạy!{Colors.END}")
        print(f"   Khởi động bằng: cd services/orchestrator && ./start_server.sh")
    
    if not (rag_ok and orchestrator_ok):
        print(f"\n{Colors.RED}Cần khởi động cả 2 services để chạy demo!{Colors.END}\n")
        return
    
    # Menu
    while True:
        print(f"\n{Colors.BOLD}Chọn demo:{Colors.END}")
        print(f"  {Colors.CYAN}1.{Colors.END} Test RAG Service trực tiếp")
        print(f"  {Colors.CYAN}2.{Colors.END} Test Agent tương tác với RAG")
        print(f"  {Colors.CYAN}3.{Colors.END} Chế độ hỏi đáp tương tác")
        print(f"  {Colors.CYAN}4.{Colors.END} Chạy tất cả demos")
        print(f"  {Colors.CYAN}0.{Colors.END} Thoát")
        
        choice = input(f"\n{Colors.BOLD}Lựa chọn (0-4): {Colors.END}").strip()
        
        if choice == "1":
            demo_rag_only()
        elif choice == "2":
            demo_orchestrator_with_rag()
        elif choice == "3":
            interactive_demo()
        elif choice == "4":
            demo_rag_only()
            demo_orchestrator_with_rag()
            
            print(f"\n{Colors.YELLOW}Bắt đầu interactive mode...{Colors.END}")
            input(f"{Colors.CYAN}Press Enter to continue...{Colors.END}")
            interactive_demo()
        elif choice == "0":
            print(f"\n{Colors.GREEN}Goodbye! 👋{Colors.END}\n")
            break
        else:
            print(f"{Colors.RED}Lựa chọn không hợp lệ!{Colors.END}")

if __name__ == "__main__":
    main()
