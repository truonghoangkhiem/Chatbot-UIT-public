"""
Recreate Weaviate collection with proper vector configuration.
This script deletes the existing collection and creates a new one with explicit vector dimensions.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.store.vector.weaviate_store import (
    get_weaviate_client,
    delete_document_collection,
    create_document_collection,
    DOCUMENT_COLLECTION
)

def main():
    """Recreate Weaviate collection with proper configuration."""
    
    print("=" * 80)
    print("RECREATING WEAVIATE COLLECTION WITH PROPER VECTOR CONFIGURATION")
    print("=" * 80)
    
    # Connect to Weaviate
    print("\n1. Connecting to Weaviate...")
    client = get_weaviate_client(url="http://localhost:8090")
    client.connect()
    print(f"✓ Connected to Weaviate")
    
    # Check if collection exists
    exists = client.collections.exists(DOCUMENT_COLLECTION)
    print(f"\n2. Collection '{DOCUMENT_COLLECTION}' exists: {exists}")
    
    # Delete if exists
    if exists:
        print(f"   Deleting existing collection...")
        success = delete_document_collection(client)
        if success:
            print(f"   ✓ Collection deleted successfully")
        else:
            print(f"   ✗ Failed to delete collection")
            return
    
    # Create new collection
    print(f"\n3. Creating new collection with proper vector configuration...")
    success = create_document_collection(client)
    
    if success:
        print(f"   ✓ Collection created successfully")
        
        # Verify collection
        collection = client.collections.get(DOCUMENT_COLLECTION)
        config = collection.config.get()
        print(f"\n4. Collection Configuration:")
        print(f"   - Name: {config.name}")
        print(f"   - Vectorizer: {config.vectorizer_config}")
        print(f"   - Vector Index: {config.vector_index_config}")
        print(f"   - Properties: {len(config.properties)} fields")
    else:
        print(f"   ✗ Failed to create collection")
    
    client.close()
    print(f"\n✓ DONE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
