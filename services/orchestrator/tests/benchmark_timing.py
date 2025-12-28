#!/usr/bin/env python3
"""
Benchmark thời gian xử lý từng phần của pipeline
"""

import asyncio
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))
rag_path = Path(__file__).parent.parent.parent / "rag_services"
sys.path.insert(0, str(rag_path))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

if not os.getenv("NEO4J_PASSWORD"):
    os.environ["NEO4J_PASSWORD"] = "password"

import logging
logging.basicConfig(level=logging.WARNING)

import httpx
from neo4j import GraphDatabase


async def benchmark_rag():
    """Benchmark RAG search"""
    print("\n📚 RAG Search:")
    start = time.time()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/search",
            json={"query": "điều kiện đăng ký học phần", "top_k": 5},
            timeout=30
        )
    elapsed = time.time() - start
    
    if response.status_code == 200:
        data = response.json()
        print(f"   Time: {elapsed:.2f}s")
        print(f"   Documents: {data.get('total_hits', 0)}")
        print(f"   Internal latency: {data.get('latency_ms', 0):.0f}ms")
    return elapsed


def benchmark_kg():
    """Benchmark KG search"""
    print("\n🔗 Knowledge Graph Search:")
    start = time.time()
    
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", os.getenv('NEO4J_PASSWORD', 'password')))
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Article)
            WHERE toLower(a.title) CONTAINS 'đăng ký' 
               OR toLower(a.full_text) CONTAINS 'đăng ký học phần'
            RETURN a.title as title
            LIMIT 5
        """)
        articles = list(result)
    driver.close()
    
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Articles: {len(articles)}")
    return elapsed


async def benchmark_llm():
    """Benchmark LLM call"""
    print("\n🤖 LLM API Call:")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("   ⚠️  No OPENROUTER_API_KEY found")
        return 3.0  # Estimate
    
    start = time.time()
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "Xin chào, bạn là ai?"}],
                "max_tokens": 100
            },
            timeout=30
        )
    elapsed = time.time() - start
    
    if response.status_code == 200:
        print(f"   Time: {elapsed:.2f}s")
    else:
        print(f"   Error: {response.status_code}")
    
    return elapsed


async def test_parallel():
    """Test parallel execution"""
    print("\n" + "="*70)
    print("🔄 PARALLEL VS SEQUENTIAL TEST")
    print("="*70)
    
    async def async_rag():
        start = time.time()
        async with httpx.AsyncClient() as client:
            await client.post(
                "http://localhost:8000/v1/search",
                json={"query": "điều kiện đăng ký học phần", "top_k": 5},
                timeout=30
            )
        return time.time() - start
    
    async def async_kg():
        start = time.time()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _sync_kg)
        return time.time() - start
    
    def _sync_kg():
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", os.getenv('NEO4J_PASSWORD', 'password')))
        with driver.session() as session:
            result = session.run("MATCH (a:Article) RETURN a.title LIMIT 5")
            list(result)
        driver.close()
    
    # Sequential
    print("\n🔀 Sequential (RAG then KG):")
    seq_start = time.time()
    rag_t = await async_rag()
    kg_t = await async_kg()
    seq_time = time.time() - seq_start
    print(f"   RAG: {rag_t:.2f}s + KG: {kg_t:.2f}s = {seq_time:.2f}s")
    
    # Parallel
    print("\n⚡ Parallel (RAG + KG together):")
    par_start = time.time()
    results = await asyncio.gather(async_rag(), async_kg())
    par_time = time.time() - par_start
    print(f"   RAG: {results[0]:.2f}s, KG: {results[1]:.2f}s (parallel) = {par_time:.2f}s")
    
    speedup = seq_time / par_time if par_time > 0 else 1
    print(f"\n📊 Speedup: {speedup:.1f}x faster with parallel")
    
    return seq_time, par_time


async def main():
    print("\n" + "🚀"*35)
    print("    PIPELINE TIMING BENCHMARK")
    print("🚀"*35)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*70)
    print("🔬 COMPONENT BENCHMARKS")
    print("="*70)
    
    # Individual components
    rag_time = await benchmark_rag()
    kg_time = benchmark_kg()
    llm_time = await benchmark_llm()
    
    print("\n" + "="*70)
    print("📊 COMPONENT SUMMARY")
    print("="*70)
    print(f"\n   RAG Search:      {rag_time:.2f}s")
    print(f"   KG Search:       {kg_time:.2f}s")
    print(f"   LLM Call:        {llm_time:.2f}s")
    print(f"   ─────────────────────────")
    
    sequential = rag_time + kg_time + llm_time * 2  # 2 LLM calls
    parallel = max(rag_time, kg_time) + llm_time * 2
    
    print(f"   If Sequential:   {sequential:.2f}s")
    print(f"   If Parallel:     {parallel:.2f}s")
    
    # Parallel test
    seq_t, par_t = await test_parallel()
    
    print("\n" + "="*70)
    print("📈 ANALYSIS")
    print("="*70)
    
    print("\n🔍 Bottleneck Analysis:")
    
    if rag_time > 1:
        print(f"   ⚠️  RAG is slow ({rag_time:.1f}s)")
        print(f"      - Weaviate vector search latency")
        print(f"      - Reranking may be adding time")
    else:
        print(f"   ✅ RAG is fast ({rag_time:.1f}s)")
    
    if kg_time > 0.5:
        print(f"   ⚠️  KG is slow ({kg_time:.1f}s)")
    else:
        print(f"   ✅ KG is fast ({kg_time:.1f}s)")
    
    if llm_time > 3:
        print(f"   ⚠️  LLM is slow ({llm_time:.1f}s)")
        print(f"      - OpenRouter API latency")
        print(f"      - Consider caching or faster model")
    else:
        print(f"   ✅ LLM is acceptable ({llm_time:.1f}s)")
    
    print(f"\n⏱️  Estimated Full Query Time:")
    print(f"   - Planning (1 LLM):     ~{llm_time:.1f}s")
    print(f"   - RAG + KG (parallel):  ~{max(rag_time, kg_time):.1f}s")
    print(f"   - Answer (1 LLM):       ~{llm_time:.1f}s")
    print(f"   ─────────────────────────────")
    print(f"   TOTAL (optimal):        ~{max(rag_time, kg_time) + llm_time * 2:.1f}s")
    
    actual_test_time = 41.68  # From earlier test
    print(f"\n   Actual observed:        ~{actual_test_time:.1f}s")
    
    if actual_test_time > parallel * 1.5:
        print(f"\n   ❌ Actual is much slower than expected!")
        print(f"      Possible issues:")
        print(f"      - RAG queries may not be parallel")
        print(f"      - Multiple LLM calls happening")
        print(f"      - Cold start / initialization overhead")
    
    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
