#!/usr/bin/env python3
"""
Interactive test script for IRCoT and Graph RAG system.

This script allows you to test the orchestrator with real queries
and see detailed flow of retrieval, reasoning, and response generation.

Usage:
    cd services/orchestrator
    python tests/test_ircot_interactive.py
"""

import asyncio
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)

# Force DEBUG logging
os.environ['LOG_LEVEL'] = 'DEBUG'

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)

# Reduce noise from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_header(title: str):
    """Print a formatted header."""
    print_separator()
    print(f"  {title}")
    print_separator()


def print_json(data: dict, indent: int = 2):
    """Print JSON data in a readable format."""
    print(json.dumps(data, indent=indent, ensure_ascii=False, default=str))


async def test_with_real_data():
    """Test the orchestrator with real data and show detailed flow."""
    from app.core.container import get_container
    from app.core.domain import OrchestrationRequest
    
    print_header("🚀 IRCoT & Graph RAG Interactive Test")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize container
    print("📦 Initializing service container...")
    container = get_container()
    
    # Get the multi-agent orchestrator
    orchestrator = container.get_multi_agent_orchestrator()
    
    print(f"✅ Orchestrator initialized")
    print(f"   - IRCoT enabled: {orchestrator.ircot_config.enabled}")
    print(f"   - IRCoT mode: {orchestrator.ircot_config.mode.value}")
    print(f"   - Max iterations: {orchestrator.ircot_config.max_iterations}")
    print(f"   - Complexity threshold: {orchestrator.ircot_config.complexity_threshold}")
    print()
    
    # Test queries
    test_queries = [
        # Simple query (should NOT trigger IRCoT)
        {
            "query": "Địa chỉ trường UIT ở đâu?",
            "expected_complexity": "simple",
            "description": "Simple factual query"
        },
        # Medium query
        {
            "query": "Điều kiện để đăng ký học phần là gì?",
            "expected_complexity": "medium",
            "description": "Standard informational query"
        },
        # Complex query (should trigger IRCoT)
        {
            "query": "Nếu sinh viên không đạt điều kiện tiên quyết của một môn học thì có những ảnh hưởng gì đến việc đăng ký các môn học khác và lộ trình học tập?",
            "expected_complexity": "complex",
            "description": "Complex multi-hop reasoning query"
        },
        # Another complex query
        {
            "query": "So sánh quy trình xin thôi học và quy trình xin bảo lưu kết quả học tập, điều kiện nào dễ hơn?",
            "expected_complexity": "complex",
            "description": "Comparative complex query"
        }
    ]
    
    # Interactive mode
    while True:
        print_header("📋 Available Test Queries")
        for i, q in enumerate(test_queries, 1):
            print(f"  [{i}] {q['description']}")
            print(f"      Query: {q['query'][:60]}...")
            print(f"      Expected: {q['expected_complexity']}")
            print()
        
        print("  [0] Enter custom query")
        print("  [q] Quit")
        print()
        
        choice = input("Select option: ").strip().lower()
        
        if choice == 'q':
            print("Goodbye!")
            break
        
        query = None
        if choice == '0':
            query = input("Enter your query: ").strip()
            if not query:
                continue
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(test_queries):
                query = test_queries[idx]["query"]
        
        if not query:
            print("Invalid choice, try again.")
            continue
        
        # Process the query
        print_header(f"🔍 Processing Query")
        print(f"Query: {query}")
        print()
        
        request = OrchestrationRequest(
            user_query=query,
            use_rag=True,
            rag_top_k=5
        )
        
        try:
            start_time = datetime.now()
            response = await orchestrator.process_request(request)
            end_time = datetime.now()
            
            # Display results
            print_header("📊 Results")
            
            # Processing stats
            stats = response.processing_stats
            print("\n🕐 Processing Statistics:")
            print(f"   - Total time: {stats.get('total_time', 0):.2f}s")
            print(f"   - Planning time: {stats.get('planning_time', 0):.2f}s")
            print(f"   - Complexity: {stats.get('plan_complexity', 'N/A')} (score: {stats.get('plan_complexity_score', 'N/A')})")
            
            # IRCoT stats (if used)
            if stats.get('ircot_mode'):
                print("\n🔄 IRCoT Statistics:")
                print(f"   - Iterations: {stats.get('ircot_iterations', 0)}")
                print(f"   - Early stopped: {stats.get('ircot_early_stopped', False)}")
                print(f"   - Confidence: {stats.get('ircot_confidence', 0):.2f}")
                print(f"   - Documents accumulated: {stats.get('ircot_documents_accumulated', 0)}")
                if stats.get('ircot_queries_used'):
                    print(f"   - Queries used:")
                    for q in stats['ircot_queries_used']:
                        print(f"      • {q[:70]}...")
            else:
                print("\n📚 Standard RAG Statistics:")
                print(f"   - RAG time: {stats.get('rag_time', 0):.2f}s")
                print(f"   - Documents retrieved: {stats.get('documents_retrieved', 0)}")
            
            # Agent metadata
            if response.agent_metadata:
                print("\n🤖 Agent Metadata:")
                if response.agent_metadata.get('plan_result'):
                    plan = response.agent_metadata['plan_result']
                    print(f"   - Intent: {plan.get('intent', 'N/A')}")
                    print(f"   - Strategy: {plan.get('strategy', 'N/A')}")
                    print(f"   - Requires RAG: {plan.get('requires_rag', 'N/A')}")
                
                if response.agent_metadata.get('ircot'):
                    print("\n   IRCoT Details:")
                    ircot = response.agent_metadata['ircot']
                    print(f"   - Final reasoning preview: {ircot.get('final_reasoning', '')[:200]}...")
            
            # RAG Context
            if response.rag_context:
                print(f"\n📄 Retrieved Documents: {len(response.rag_context.retrieved_documents)}")
                for i, doc in enumerate(response.rag_context.retrieved_documents[:3], 1):
                    title = doc.get('title', f'Document {i}')
                    score = doc.get('score', 0)
                    content_preview = doc.get('content', '')[:100]
                    print(f"\n   [{i}] {title} (score: {score:.3f})")
                    print(f"       {content_preview}...")
            
            # Final response
            print_header("💬 Final Response")
            print(response.response)
            print()
            
        except Exception as e:
            logger.exception(f"Error processing query: {e}")
            print(f"\n❌ Error: {e}")
        
        print()
        input("Press Enter to continue...")
        print("\n" * 2)


async def quick_test():
    """Quick test with a single complex query."""
    from app.core.container import get_container
    from app.core.domain import OrchestrationRequest
    
    print_header("🚀 Quick IRCoT Test")
    
    container = get_container()
    orchestrator = container.get_multi_agent_orchestrator()
    
    # Complex query to trigger IRCoT
    query = "Điều kiện để được xét tốt nghiệp sớm là gì và cần chuẩn bị những giấy tờ gì?"
    
    print(f"Query: {query}")
    print()
    
    request = OrchestrationRequest(
        user_query=query,
        use_rag=True,
        rag_top_k=5
    )
    
    response = await orchestrator.process_request(request)
    
    print_header("Results")
    print(f"IRCoT mode: {response.processing_stats.get('ircot_mode', False)}")
    print(f"Iterations: {response.processing_stats.get('ircot_iterations', 0)}")
    print(f"Total time: {response.processing_stats.get('total_time', 0):.2f}s")
    print()
    print("Response:")
    print(response.response)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test IRCoT and Graph RAG system")
    parser.add_argument("--quick", action="store_true", help="Run quick test instead of interactive mode")
    args = parser.parse_args()
    
    if args.quick:
        asyncio.run(quick_test())
    else:
        asyncio.run(test_with_real_data())
