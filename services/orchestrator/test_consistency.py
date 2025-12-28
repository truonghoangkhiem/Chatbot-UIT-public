#!/usr/bin/env python3
"""Test RAG query consistency"""
import requests
import time

query = "quy định về tiếng anh của trường thế nào"
url = "http://localhost:8001/api/v1/chat"

print(f"Testing query: {query}")
print("=" * 60)

for i in range(3):
    print(f"\n--- Test {i+1} ---")
    try:
        start = time.time()
        response = requests.post(url, json={"query": query}, timeout=60)
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            resp_text = data.get('response', '')[:200]
            rag_ctx = data.get('rag_context', {})
            docs_count = rag_ctx.get('total_documents', 0)
            search_mode = rag_ctx.get('search_mode', 'unknown')
            
            print(f"Status: OK ({elapsed:.2f}s)")
            print(f"Documents: {docs_count}, Mode: {search_mode}")
            print(f"Response preview: {resp_text}...")
        else:
            print(f"Status: {response.status_code}")
            print(f"Error: {response.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(1)

print("\n" + "=" * 60)
print("Done!")
