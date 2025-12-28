#!/usr/bin/env python3
"""
Interactive IRCoT Test Script with Real Data.

This script tests the IRCoT (Interleaving Retrieval with Chain-of-Thought) 
system with real queries and shows detailed flow information.

Usage:
    python test_ircot_realtime.py

Requirements:
    - Orchestrator service running at localhost:8001
    - RAG service running at localhost:8000
    - LOG_LEVEL=DEBUG in orchestrator .env
"""

import requests
import json
import time
from datetime import datetime

# Configuration
ORCHESTRATOR_URL = "http://localhost:8001"
RAG_URL = "http://localhost:8000"

# Test queries - mix of simple and complex
TEST_QUERIES = {
    "simple": [
        "Địa chỉ trường UIT ở đâu?",
        "Điểm tối đa của một môn học là bao nhiêu?",
    ],
    "complex": [
        "Nếu tôi không đạt điều kiện tiên quyết của môn SE101 thì những môn học nào sẽ bị ảnh hưởng?",
        "So sánh điều kiện tốt nghiệp giữa chương trình đại trà và chương trình chất lượng cao?",
        "Sinh viên bị cảnh báo học vụ khi nào và cần làm gì để thoát khỏi tình trạng này?",
        "Điều kiện để được xét tốt nghiệp sớm là gì và cần chuẩn bị những gì?",
    ]
}


def check_services():
    """Check if services are running."""
    print("=" * 70)
    print("🔍 CHECKING SERVICES")
    print("=" * 70)
    
    services_ok = True
    
    # Check RAG service
    try:
        resp = requests.get(f"{RAG_URL}/health", timeout=5)
        if resp.status_code == 200:
            print(f"✅ RAG Service: OK ({RAG_URL})")
        else:
            print(f"⚠️ RAG Service: Status {resp.status_code}")
    except Exception as e:
        print(f"❌ RAG Service: FAILED - {e}")
        services_ok = False
    
    # Check Orchestrator service
    try:
        resp = requests.get(f"{ORCHESTRATOR_URL}/api/v1/health", timeout=5)
        data = resp.json()
        status = data.get("status", "unknown")
        ircot = data.get("services", {}).get("ircot_enabled", "unknown")
        print(f"✅ Orchestrator Service: {status} ({ORCHESTRATOR_URL})")
        print(f"   IRCoT Enabled: {ircot}")
    except Exception as e:
        print(f"❌ Orchestrator Service: FAILED - {e}")
        services_ok = False
    
    print()
    return services_ok


def send_query(query: str, session_id: str = None) -> dict:
    """Send a query to the orchestrator."""
    payload = {
        "message": query,
        "session_id": session_id or f"test_{int(time.time())}",
        "use_rag": True,
        "rag_top_k": 5
    }
    
    try:
        resp = requests.post(
            f"{ORCHESTRATOR_URL}/api/v1/chat",
            json=payload,
            timeout=120  # Long timeout for complex queries
        )
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def print_response_details(response: dict, query: str):
    """Print detailed response information."""
    print("\n" + "=" * 70)
    print(f"📝 QUERY: {query}")
    print("=" * 70)
    
    if "error" in response:
        print(f"❌ ERROR: {response['error']}")
        return
    
    # Basic response info
    print(f"\n📤 RESPONSE:")
    print("-" * 50)
    answer = response.get("response", "No response")
    print(f"{answer[:500]}..." if len(answer) > 500 else answer)
    
    # Processing stats
    stats = response.get("processing_stats", {})
    print(f"\n📊 PROCESSING STATS:")
    print("-" * 50)
    
    # Check if IRCoT was used
    ircot_mode = stats.get("ircot_mode", False)
    if ircot_mode:
        print(f"🔄 IRCoT MODE: ENABLED")
        print(f"   Iterations: {stats.get('ircot_iterations', 'N/A')}")
        print(f"   Documents Accumulated: {stats.get('ircot_documents_accumulated', 'N/A')}")
        print(f"   Confidence: {stats.get('ircot_confidence', 'N/A')}")
        print(f"   Early Stopped: {stats.get('ircot_early_stopped', 'N/A')}")
        print(f"   IRCoT Time: {stats.get('ircot_time', 'N/A'):.2f}s" if stats.get('ircot_time') else "")
        
        # Show queries used
        queries_used = stats.get('ircot_queries_used', [])
        if queries_used:
            print(f"\n   🔍 Queries Used:")
            for i, q in enumerate(queries_used, 1):
                print(f"      {i}. {q}")
    else:
        print(f"📋 Standard RAG Mode")
        print(f"   Documents Retrieved: {stats.get('documents_retrieved', 'N/A')}")
    
    # Timing info
    print(f"\n⏱️ TIMING:")
    print(f"   Planning: {stats.get('planning_time', 'N/A'):.3f}s" if stats.get('planning_time') else "   Planning: N/A")
    print(f"   RAG: {stats.get('rag_time', 'N/A'):.3f}s" if stats.get('rag_time') else "   RAG: N/A")
    print(f"   Answer Gen: {stats.get('answer_generation_time', 'N/A'):.3f}s" if stats.get('answer_generation_time') else "   Answer Gen: N/A")
    print(f"   Formatting: {stats.get('formatting_time', 'N/A'):.3f}s" if stats.get('formatting_time') else "   Formatting: N/A")
    print(f"   Total: {stats.get('total_time', 'N/A'):.3f}s" if stats.get('total_time') else "   Total: N/A")
    
    # Complexity info
    complexity = stats.get('plan_complexity', stats.get('complexity', 'N/A'))
    complexity_score = stats.get('plan_complexity_score', stats.get('complexity_score', 'N/A'))
    print(f"\n🎯 COMPLEXITY:")
    print(f"   Level: {complexity}")
    print(f"   Score: {complexity_score}")
    
    # Agent metadata
    agent_meta = response.get("agent_metadata", {})
    if agent_meta:
        print(f"\n🤖 AGENT METADATA:")
        print(f"   Pipeline: {agent_meta.get('pipeline', 'N/A')}")
        print(f"   Answer Confidence: {agent_meta.get('answer_confidence', 'N/A')}")
        
        # IRCoT details
        ircot_data = agent_meta.get('ircot', {})
        if ircot_data:
            print(f"\n   📋 IRCoT Details:")
            print(f"      Final Reasoning: {ircot_data.get('final_reasoning', 'N/A')[:200]}...")
            iterations = ircot_data.get('iterations', [])
            if iterations:
                print(f"      Iterations:")
                for it in iterations:
                    print(f"         - Step {it.get('iteration')}: conf={it.get('confidence')}, can_answer={it.get('can_answer_now')}")
    
    # RAG Context info
    rag_context = response.get("rag_context", {})
    if rag_context:
        docs = rag_context.get("retrieved_documents", [])
        print(f"\n📚 RAG CONTEXT:")
        print(f"   Total Documents: {len(docs)}")
        if docs:
            print(f"   Top 3 Documents:")
            for i, doc in enumerate(docs[:3], 1):
                title = doc.get("title", "Unknown")
                score = doc.get("score", 0)
                print(f"      {i}. [{score:.3f}] {title}")


def run_interactive_test():
    """Run interactive test mode."""
    print("\n" + "=" * 70)
    print("🚀 INTERACTIVE IRCoT TEST")
    print("=" * 70)
    print("Type your question or use these commands:")
    print("  /simple  - Run simple test queries")
    print("  /complex - Run complex test queries")
    print("  /all     - Run all test queries")
    print("  /quit    - Exit")
    print()
    
    session_id = f"interactive_{int(time.time())}"
    
    while True:
        try:
            query = input("\n💬 Your question: ").strip()
            
            if not query:
                continue
            
            if query.lower() == "/quit":
                print("Goodbye! 👋")
                break
            
            if query.lower() == "/simple":
                for q in TEST_QUERIES["simple"]:
                    response = send_query(q, session_id)
                    print_response_details(response, q)
                continue
            
            if query.lower() == "/complex":
                for q in TEST_QUERIES["complex"]:
                    response = send_query(q, session_id)
                    print_response_details(response, q)
                continue
            
            if query.lower() == "/all":
                for category, queries in TEST_QUERIES.items():
                    print(f"\n{'='*70}")
                    print(f"📂 CATEGORY: {category.upper()}")
                    print(f"{'='*70}")
                    for q in queries:
                        response = send_query(q, session_id)
                        print_response_details(response, q)
                continue
            
            # Regular query
            print("\n⏳ Processing...")
            start = time.time()
            response = send_query(query, session_id)
            elapsed = time.time() - start
            print(f"✅ Completed in {elapsed:.2f}s")
            print_response_details(response, query)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("🔬 IRCoT REAL-TIME TEST SYSTEM")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Check services
    if not check_services():
        print("❌ Some services are not available. Please start them first.")
        print("\nTo start services:")
        print("  cd /home/kien/Kien/study/SE363.Q11/Chatbot-UIT")
        print("  python start_backend.py")
        return
    
    # Run interactive test
    run_interactive_test()


if __name__ == "__main__":
    main()
