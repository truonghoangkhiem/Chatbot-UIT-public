"""
Detailed timing benchmark for each component of the RAG pipeline.
This helps identify exact bottlenecks.
"""

import asyncio
import aiohttp
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase

# Configuration
RAG_SERVICE_URL = "http://localhost:8000"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')


async def benchmark_single_rag_query(query: str, use_rerank: bool = True):
    """Benchmark a single RAG query."""
    payload = {
        "query": query,
        "top_k": 5,
        "search_mode": "hybrid",
        "use_rerank": use_rerank,
        "rerank_top_n": 5,
        "need_citation": True,
        "include_char_spans": True,
        "highlight_matches": True
    }
    
    async with aiohttp.ClientSession() as session:
        start = time.time()
        async with session.post(
            f"{RAG_SERVICE_URL}/v1/search",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status == 200:
                data = await response.json()
                elapsed = time.time() - start
                return {
                    "success": True,
                    "time": elapsed,
                    "hits": len(data.get("hits", [])),
                    "query": query[:50]
                }
            else:
                return {"success": False, "status": response.status}


async def benchmark_parallel_rag_queries(queries: list, use_rerank: bool = True):
    """Benchmark multiple RAG queries in parallel."""
    print(f"\n📊 Testing {len(queries)} queries in PARALLEL (rerank={use_rerank})...")
    
    async with aiohttp.ClientSession() as session:
        async def single_query(query):
            payload = {
                "query": query,
                "top_k": 5,
                "search_mode": "hybrid",
                "use_rerank": use_rerank,
                "rerank_top_n": 5,
                "need_citation": True,
                "include_char_spans": True
            }
            start = time.time()
            async with session.post(
                f"{RAG_SERVICE_URL}/v1/search",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    await response.json()
                    return time.time() - start
                return None
        
        total_start = time.time()
        results = await asyncio.gather(*[single_query(q) for q in queries])
        total_time = time.time() - total_start
        
        return {
            "total_time": total_time,
            "individual_times": results,
            "count": len(queries)
        }


async def benchmark_sequential_rag_queries(queries: list, use_rerank: bool = True):
    """Benchmark multiple RAG queries sequentially."""
    print(f"\n📊 Testing {len(queries)} queries SEQUENTIALLY (rerank={use_rerank})...")
    
    async with aiohttp.ClientSession() as session:
        total_start = time.time()
        individual_times = []
        
        for query in queries:
            payload = {
                "query": query,
                "top_k": 5,
                "search_mode": "hybrid",
                "use_rerank": use_rerank,
                "rerank_top_n": 5,
                "need_citation": True,
                "include_char_spans": True
            }
            start = time.time()
            async with session.post(
                f"{RAG_SERVICE_URL}/v1/search",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    await response.json()
                    individual_times.append(time.time() - start)
                else:
                    individual_times.append(None)
        
        total_time = time.time() - total_start
        
        return {
            "total_time": total_time,
            "individual_times": individual_times,
            "count": len(queries)
        }


def benchmark_kg_query(query: str):
    """Benchmark Neo4j query."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        with driver.session() as session:
            # Full-text search
            start = time.time()
            result = session.run("""
                CALL db.index.fulltext.queryNodes('node_name_content', $query) 
                YIELD node, score
                RETURN node.name as name, labels(node) as labels, score
                LIMIT 10
            """, query=query)
            records = list(result)
            ft_time = time.time() - start
            
            # Path query
            start = time.time()
            result = session.run("""
                MATCH (n)-[r]->(m)
                WHERE n.name CONTAINS 'Điều 14' OR n.name CONTAINS 'thừa kế'
                RETURN n.name, type(r), m.name
                LIMIT 20
            """)
            records2 = list(result)
            path_time = time.time() - start
            
            return {
                "fulltext_time": ft_time,
                "path_time": path_time,
                "fulltext_results": len(records),
                "path_results": len(records2)
            }
    finally:
        driver.close()


async def main():
    print("="*60)
    print("🔬 DETAILED TIMING BENCHMARK")
    print("="*60)
    
    # Test queries (similar to what IRCoT generates)
    test_queries = [
        "người thừa kế theo Điều 14 quy chế đào tạo",
        "điều kiện xét tốt nghiệp đại học",
        "quy định về miễn giảm học phí",
        "thời gian đào tạo chương trình cử nhân",
        "điểm trung bình tích lũy tối thiểu"
    ]
    
    # 1. Single RAG query
    print("\n" + "="*60)
    print("1️⃣ SINGLE RAG QUERY")
    print("="*60)
    
    result = await benchmark_single_rag_query(test_queries[0], use_rerank=True)
    print(f"   Query: {result['query']}...")
    print(f"   Time: {result['time']:.2f}s")
    print(f"   Hits: {result.get('hits', 0)}")
    
    # 2. Single RAG query WITHOUT reranking
    print("\n" + "="*60)
    print("2️⃣ SINGLE RAG QUERY (NO RERANK)")
    print("="*60)
    
    result = await benchmark_single_rag_query(test_queries[0], use_rerank=False)
    print(f"   Query: {result['query']}...")
    print(f"   Time: {result['time']:.2f}s")
    print(f"   Hits: {result.get('hits', 0)}")
    
    # 3. Parallel RAG queries
    print("\n" + "="*60)
    print("3️⃣ PARALLEL RAG QUERIES (5 queries)")
    print("="*60)
    
    parallel_with_rerank = await benchmark_parallel_rag_queries(test_queries, use_rerank=True)
    print(f"   Total time: {parallel_with_rerank['total_time']:.2f}s")
    print(f"   Individual times: {[f'{t:.2f}s' if t else 'fail' for t in parallel_with_rerank['individual_times']]}")
    
    parallel_no_rerank = await benchmark_parallel_rag_queries(test_queries, use_rerank=False)
    print(f"\n   Without rerank:")
    print(f"   Total time: {parallel_no_rerank['total_time']:.2f}s")
    print(f"   Individual times: {[f'{t:.2f}s' if t else 'fail' for t in parallel_no_rerank['individual_times']]}")
    
    # 4. Sequential RAG queries (for comparison)
    print("\n" + "="*60)
    print("4️⃣ SEQUENTIAL RAG QUERIES (5 queries)")
    print("="*60)
    
    sequential_with_rerank = await benchmark_sequential_rag_queries(test_queries, use_rerank=True)
    print(f"   Total time: {sequential_with_rerank['total_time']:.2f}s")
    print(f"   Individual times: {[f'{t:.2f}s' if t else 'fail' for t in sequential_with_rerank['individual_times']]}")
    
    # 5. KG queries
    print("\n" + "="*60)
    print("5️⃣ KNOWLEDGE GRAPH QUERIES")
    print("="*60)
    
    kg_result = benchmark_kg_query("thừa kế Điều 14")
    print(f"   Fulltext search: {kg_result['fulltext_time']:.3f}s ({kg_result['fulltext_results']} results)")
    print(f"   Path traversal: {kg_result['path_time']:.3f}s ({kg_result['path_results']} results)")
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    print(f"""
   Single RAG (with rerank):  ~{parallel_with_rerank['individual_times'][0]:.2f}s per query
   Single RAG (no rerank):    ~{parallel_no_rerank['individual_times'][0]:.2f}s per query
   
   5 Parallel (with rerank):  {parallel_with_rerank['total_time']:.2f}s total
   5 Parallel (no rerank):    {parallel_no_rerank['total_time']:.2f}s total
   5 Sequential (with rerank): {sequential_with_rerank['total_time']:.2f}s total
   
   KG Fulltext: {kg_result['fulltext_time']:.3f}s
   KG Path:     {kg_result['path_time']:.3f}s
   
   ⚡ Rerank impact: {(parallel_with_rerank['total_time'] / max(parallel_no_rerank['total_time'], 0.01)):.1f}x slower
   ⚡ Parallel speedup: {(sequential_with_rerank['total_time'] / max(parallel_with_rerank['total_time'], 0.01)):.1f}x faster
""")
    
    # Calculate expected pipeline time
    print("\n" + "="*60)
    print("⏱️  EXPECTED PIPELINE TIME (2-agent)")
    print("="*60)
    
    llm_time = 2.5  # From previous benchmark
    rag_time = parallel_with_rerank['total_time']
    kg_time = kg_result['fulltext_time'] + kg_result['path_time']
    
    # Stage 1: Planning (1 LLM call)
    stage1 = llm_time
    
    # Stage 2: RAG + KG in parallel
    stage2 = max(rag_time, kg_time)
    
    # Stage 3: Answer generation (1 LLM call)
    stage3 = llm_time
    
    expected_total = stage1 + stage2 + stage3
    
    print(f"""
   Stage 1 (Planning):        {stage1:.2f}s (1 LLM call)
   Stage 2 (RAG + KG):        {stage2:.2f}s (parallel)
       - RAG ({len(test_queries)} queries): {rag_time:.2f}s
       - KG:               {kg_time:.3f}s
   Stage 3 (Answer):          {stage3:.2f}s (1 LLM call)
   
   ═══════════════════════════════════════
   EXPECTED TOTAL:            ~{expected_total:.1f}s
   ═══════════════════════════════════════
""")


if __name__ == "__main__":
    asyncio.run(main())
