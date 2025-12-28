#!/usr/bin/env python3
"""Debug RAG retrieved documents"""
import requests

query = "quy định về tiếng anh của trường thế nào"
url = "http://localhost:8001/api/v1/chat"

print(f"Query: {query}")
print("=" * 60)

response = requests.post(url, json={"query": query}, timeout=60)

if response.status_code == 200:
    data = response.json()
    rag_ctx = data.get('rag_context', {})
    docs = rag_ctx.get('documents', [])
    
    print(f"\nTotal documents retrieved: {len(docs)}")
    print(f"Search mode: {rag_ctx.get('search_mode', 'unknown')}")
    print(f"Use KG: {rag_ctx.get('use_knowledge_graph')}")
    print(f"Use Vector: {rag_ctx.get('use_vector_search')}")
    print(f"Strategy: {rag_ctx.get('strategy')}")
    
    print("\n" + "=" * 60)
    print("RETRIEVED DOCUMENTS:")
    print("=" * 60)
    
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        if isinstance(doc, dict):
            title = doc.get('title', doc.get('metadata', {}).get('title', 'N/A'))
            text = doc.get('text', doc.get('content', doc.get('page_content', '')))[:300]
            source = doc.get('source', doc.get('metadata', {}).get('source', 'N/A'))
            score = doc.get('score', doc.get('_score', 'N/A'))
            print(f"Title: {title}")
            print(f"Source: {source}")
            print(f"Score: {score}")
            print(f"Content: {text}...")
        else:
            print(f"Raw: {str(doc)[:300]}...")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
