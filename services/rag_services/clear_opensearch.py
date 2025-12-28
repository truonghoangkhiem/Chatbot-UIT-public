#!/usr/bin/env python3
"""Script to clear all data from OpenSearch index"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from infrastructure.store.opensearch.client import OpenSearchClient

def main():
    print("Connecting to OpenSearch...")
    client = OpenSearchClient()
    
    # Count documents before deletion
    try:
        count_result = client.client.count(index=client.index_name)
        doc_count = count_result.get("count", 0)
        print(f"Current document count: {doc_count}")
    except Exception as e:
        print(f"Error counting documents: {e}")
        doc_count = "unknown"
    
    # Delete all documents
    print(f"Deleting all documents from index: {client.index_name}")
    
    result = client.client.delete_by_query(
        index=client.index_name,
        body={"query": {"match_all": {}}}
    )
    
    deleted = result.get("deleted", 0)
    print(f"Successfully deleted {deleted} documents")
    
    # Refresh index
    client.client.indices.refresh(index=client.index_name)
    print("Index refreshed")

if __name__ == "__main__":
    main()
