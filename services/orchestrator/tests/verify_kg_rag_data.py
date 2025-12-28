#!/usr/bin/env python3
"""
Kiểm tra dữ liệu thực tế trong KG và RAG
Xác minh câu trả lời của LLM có đúng không
"""

import os
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))
rag_path = Path(__file__).parent.parent.parent / "rag_services"
sys.path.insert(0, str(rag_path))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

if not os.getenv("NEO4J_PASSWORD"):
    os.environ["NEO4J_PASSWORD"] = "password"

print("="*70)
print("KIỂM TRA DỮ LIỆU THỰC TẾ - Điều kiện đăng ký học phần")
print("="*70)

# ============================================
# 1. KIỂM TRA KNOWLEDGE GRAPH
# ============================================
print("\n" + "="*70)
print("📊 PHẦN 1: KNOWLEDGE GRAPH (Neo4j)")
print("="*70)

from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", os.getenv('NEO4J_PASSWORD', 'password')))

with driver.session() as session:
    # Tìm Điều 14 (về đăng ký học phần)
    result = session.run("""
        MATCH (a:Article)
        WHERE a.title CONTAINS 'Điều 14' OR a.title CONTAINS 'đăng ký'
        RETURN a.title as title, a.full_text as content
        LIMIT 3
    """)
    
    print("\n📚 CÁC ĐIỀU KHOẢN VỀ ĐĂNG KÝ HỌC PHẦN:")
    print("-"*70)
    
    articles_found = []
    for record in result:
        articles_found.append(record)
        title = record["title"]
        content = record["content"] or ""
        print(f"\n✅ {title}")
        print("-"*50)
        # In ra nội dung đầy đủ hơn
        if content:
            print(content[:1500])
            if len(content) > 1500:
                print("...")
    
    if not articles_found:
        print("❌ Không tìm thấy điều khoản nào!")

driver.close()

# ============================================
# 2. KIỂM TRA RAG (Vector Search)
# ============================================
print("\n" + "="*70)
print("📊 PHẦN 2: RAG VECTOR SEARCH")
print("="*70)

import httpx

# Gọi RAG API để tìm kiếm
try:
    response = httpx.post(
        "http://localhost:8000/api/v1/search",
        json={
            "query": "điều kiện đăng ký học phần tín chỉ",
            "top_k": 5
        },
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        
        print(f"\n📚 TÌM THẤY {len(results)} DOCUMENTS:")
        print("-"*70)
        
        for i, doc in enumerate(results, 1):
            score = doc.get("score", 0)
            text = doc.get("text", doc.get("content", ""))[:800]
            source = doc.get("metadata", {}).get("source", "Unknown")
            
            print(f"\n[{i}] Score: {score:.4f}")
            print(f"    Source: {source}")
            print(f"    Content: {text}")
            if len(doc.get("text", doc.get("content", ""))) > 800:
                print("    ...")
    else:
        print(f"❌ RAG API error: {response.status_code}")
        print(response.text)

except Exception as e:
    print(f"❌ Error calling RAG API: {e}")

# ============================================
# 3. XÁC MINH CÂU TRẢ LỜI CỦA LLM
# ============================================
print("\n" + "="*70)
print("📊 PHẦN 3: XÁC MINH CÂU TRẢ LỜI LLM")
print("="*70)

llm_claims = [
    ("14-24 tín chỉ trong học kỳ chính", "14", "24"),
    ("Tối đa 30 tín chỉ nếu ĐTBC >= 8.0", "30", "8.0"),
    ("Học kỳ hè tối đa 12 tín chỉ", "12", "hè"),
    ("Điểm dưới 5.0 phải học lại", "5.0", "học lại"),
]

print("\nKiểm tra các thông tin LLM đã trả lời:")
print("-"*70)

# Đọc lại từ KG để verify
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", os.getenv('NEO4J_PASSWORD', 'password')))

with driver.session() as session:
    result = session.run("""
        MATCH (a:Article)
        WHERE a.title CONTAINS 'Điều 14' 
           OR a.title CONTAINS 'Điều 15'
           OR a.title CONTAINS 'đăng ký'
           OR toLower(a.full_text) CONTAINS 'tín chỉ'
        RETURN a.title as title, a.full_text as content
    """)
    
    all_content = ""
    for record in result:
        all_content += (record["content"] or "") + " "
    
    for claim, keyword1, keyword2 in llm_claims:
        found1 = keyword1 in all_content
        found2 = keyword2 in all_content
        status = "✅" if (found1 and found2) else "⚠️"
        print(f"{status} '{claim}'")
        print(f"   - '{keyword1}' found: {found1}")
        print(f"   - '{keyword2}' found: {found2}")

driver.close()

print("\n" + "="*70)
print("✅ KIỂM TRA HOÀN TẤT")
print("="*70)
