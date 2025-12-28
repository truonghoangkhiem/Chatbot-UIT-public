#!/usr/bin/env python3
"""Search OpenSearch for specific content"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from infrastructure.store.opensearch.client import OpenSearchClient

def search_content(query: str, size: int = 5):
    client = OpenSearchClient()
    
    result = client.client.search(
        index=client.index_name,
        body={
            'query': {
                'multi_match': {
                    'query': query,
                    'fields': ['text', 'title'],
                    'type': 'best_fields'
                }
            },
            'size': size
        }
    )
    
    print(f"Query: '{query}'")
    print(f"Found {result['hits']['total']['value']} documents\n")
    
    for i, hit in enumerate(result['hits']['hits'], 1):
        print(f"--- Result {i} (Score: {hit['_score']:.2f}) ---")
        print(f"Title: {hit['_source'].get('title', 'N/A')}")
        print(f"Chunk ID: {hit['_source'].get('chunk_id', 'N/A')}")
        text = hit['_source'].get('text', '')[:300]
        print(f"Text: {text}...")
        print()

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "kỷ luật vi phạm chuyển trường"
    search_content(query)
