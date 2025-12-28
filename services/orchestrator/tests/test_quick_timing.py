"""
Quick test to count actual RAG queries and measure real timing.
"""

import asyncio
import time
import sys
import os

# Set DEBUG log level
os.environ["LOG_LEVEL"] = "DEBUG"

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from app.core.container import get_container
from app.core.domain import OrchestrationRequest

async def main():
    print("="*70)
    print("🔬 REAL QUERY TEST WITH DETAILED TIMING")
    print("="*70)
    
    container = get_container()
    orchestrator = container.get_multi_agent_orchestrator()
    
    query = "Điều 14 quy định gì về thừa kế?"
    
    print(f"\n📝 Query: {query}")
    print("\n" + "-"*70)
    
    request = OrchestrationRequest(
        user_query=query,
        rag_top_k=5,
        use_knowledge_graph=True
    )
    
    # Time the request
    start_time = time.time()
    result = await orchestrator.process(request)
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    
    # Print timing from stats
    stats = result.processing_stats
    print("\n⏱️  TIMING BREAKDOWN:")
    for key, value in stats.items():
        if 'time' in key.lower():
            print(f"   {key}: {value:.2f}s" if isinstance(value, (int, float)) else f"   {key}: {value}")
    
    print(f"\n   TOTAL TIME: {total_time:.2f}s")
    
    # Print RAG stats
    print("\n📊 RAG STATS:")
    for key, value in stats.items():
        if 'rag' in key.lower() or 'doc' in key.lower() or 'queries' in key.lower():
            print(f"   {key}: {value}")
    
    # Print graph stats
    print("\n📊 GRAPH STATS:")
    for key, value in stats.items():
        if 'graph' in key.lower() or 'node' in key.lower() or 'path' in key.lower():
            print(f"   {key}: {value}")
    
    # Print answer
    print(f"\n💬 ANSWER (first 300 chars):")
    print(result.response[:300] if result.response else "No response")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(main())
