#!/usr/bin/env python3
"""Check OpenSearch search for English regulations"""
from infrastructure.store.opensearch.client import OpenSearchClient

client = OpenSearchClient()

print("=== Test OpenSearch Search ===")
query = "quy định tiếng anh"
print(f"Query: {query}")

# Search
result = client.search(query, size=5)
print(f"\nKết quả ({len(result)} documents):")
for i, doc in enumerate(result[:5]):
    title = doc.get('title', 'N/A')[:60]
    text = doc.get('text', doc.get('content', ''))[:150]
    score = doc.get('score', doc.get('_score', 0))
    print(f"\n{i+1}. Score: {score}")
    print(f"   Title: {title}")
    print(f"   Text: {text}...")
