#!/usr/bin/env python3
"""
Script to clear ALL data from Weaviate, Neo4j, and OpenSearch.
Use this to start fresh with clean databases.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def clear_weaviate():
    """Clear all data from Weaviate."""
    print("\n" + "="*60)
    print("🔵 CLEARING WEAVIATE")
    print("="*60)
    
    try:
        from infrastructure.store.vector.weaviate_store import get_weaviate_client, DOCUMENT_COLLECTION
        from weaviate.classes.query import Filter
        
        client = get_weaviate_client("http://localhost:8090")
        
        # Check if collection exists
        if client.collections.exists(DOCUMENT_COLLECTION):
            collection = client.collections.get(DOCUMENT_COLLECTION)
            
            # Count objects before deletion
            count_result = collection.aggregate.over_all(total_count=True)
            total_count = count_result.total_count if count_result else 0
            print(f"📊 Current objects in '{DOCUMENT_COLLECTION}': {total_count}")
            
            if total_count > 0:
                # Delete collection and recreate it (cleanest approach)
                print(f"🗑️  Deleting collection '{DOCUMENT_COLLECTION}'...")
                client.collections.delete(DOCUMENT_COLLECTION)
                print(f"✅ Deleted collection '{DOCUMENT_COLLECTION}'")
                
                # Note: Collection will be recreated on next import
                print("ℹ️  Collection will be recreated on next data import")
            else:
                print("✅ Collection already empty")
        else:
            print(f"⚠️  Collection '{DOCUMENT_COLLECTION}' does not exist")
        
        client.close()
        print("✅ Weaviate cleared successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error clearing Weaviate: {e}")
        return False


def clear_neo4j():
    """Clear all data from Neo4j."""
    print("\n" + "="*60)
    print("🟢 CLEARING NEO4J")
    print("="*60)
    
    try:
        from neo4j import GraphDatabase
        import os
        
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            # Count current nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            total = result.single()["count"]
            print(f"📊 Current nodes: {total}")
            
            if total > 0:
                # Delete all nodes and relationships
                session.run("MATCH (n) DETACH DELETE n")
                print(f"✅ Deleted {total} nodes and their relationships")
            else:
                print("✅ Database already empty")
        
        driver.close()
        print("✅ Neo4j cleared successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error clearing Neo4j: {e}")
        return False


def clear_opensearch():
    """Clear all data from OpenSearch."""
    print("\n" + "="*60)
    print("🟡 CLEARING OPENSEARCH")
    print("="*60)
    
    try:
        from infrastructure.store.opensearch.client import OpenSearchClient
        
        client = OpenSearchClient()
        
        # Count documents before deletion
        try:
            count_result = client.client.count(index=client.index_name)
            doc_count = count_result.get("count", 0)
            print(f"📊 Current documents in '{client.index_name}': {doc_count}")
        except Exception:
            doc_count = 0
            print("📊 Index may not exist or is empty")
        
        if doc_count > 0:
            # Delete all documents
            result = client.client.delete_by_query(
                index=client.index_name,
                body={"query": {"match_all": {}}}
            )
            deleted = result.get("deleted", 0)
            print(f"✅ Deleted {deleted} documents")
            
            # Refresh index
            client.client.indices.refresh(index=client.index_name)
        else:
            print("✅ Index already empty")
        
        print("✅ OpenSearch cleared successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error clearing OpenSearch: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("🧹 CLEARING ALL DATABASES")
    print("="*60)
    print("\n⚠️  This will DELETE ALL data from:")
    print("   - Weaviate (Vector Store)")
    print("   - Neo4j (Graph Database)")
    print("   - OpenSearch (Text Search)")
    
    response = input("\n❓ Are you sure? (y/yes): ")
    
    if response.lower() not in ('yes', 'y'):
        print("\n❌ Cancelled.")
        return
    
    print("\n🚀 Starting cleanup...\n")
    
    results = {
        "Weaviate": clear_weaviate(),
        "Neo4j": clear_neo4j(),
        "OpenSearch": clear_opensearch()
    }
    
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    
    for db, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"   {db}: {status}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\n🎉 All databases cleared successfully!")
    else:
        print("\n⚠️  Some databases failed to clear. Check errors above.")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
