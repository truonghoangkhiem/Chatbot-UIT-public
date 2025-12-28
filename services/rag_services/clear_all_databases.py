"""
Script to clear all data from Weaviate, OpenSearch, and Neo4j databases.
"""
import os
from dotenv import load_dotenv

load_dotenv()

def clear_weaviate():
    """Clear all classes from Weaviate."""
    print("=== Clearing Weaviate ===")
    try:
        import weaviate
        url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        client = weaviate.Client(url)
        schema = client.schema.get()
        classes = schema.get("classes", [])
        
        if not classes:
            print("Weaviate: No classes found")
            return
            
        for cls in classes:
            client.schema.delete_class(cls["class"])
            print(f"  Deleted class: {cls['class']}")
        print(f"Weaviate: Cleared {len(classes)} classes!")
    except Exception as e:
        print(f"Weaviate error: {e}")


def clear_opensearch():
    """Clear all indices from OpenSearch."""
    print("\n=== Clearing OpenSearch ===")
    try:
        from opensearchpy import OpenSearch
        
        host = os.getenv("OPENSEARCH_HOST", "localhost")
        port = int(os.getenv("OPENSEARCH_PORT", "9200"))
        user = os.getenv("OPENSEARCH_USER", "admin")
        password = os.getenv("OPENSEARCH_PASSWORD", "Admin@123")
        
        client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=(user, password),
            use_ssl=False,
            verify_certs=False,
            ssl_show_warn=False
        )
        
        indices = list(client.indices.get_alias("*").keys())
        deleted = 0
        
        for idx in indices:
            if not idx.startswith("."):  # Skip system indices
                client.indices.delete(index=idx)
                print(f"  Deleted index: {idx}")
                deleted += 1
        
        if deleted == 0:
            print("OpenSearch: No user indices found")
        else:
            print(f"OpenSearch: Cleared {deleted} indices!")
    except Exception as e:
        print(f"OpenSearch error: {e}")


def clear_neo4j():
    """Clear all nodes and relationships from Neo4j."""
    print("\n=== Clearing Neo4j ===")
    try:
        from neo4j import GraphDatabase
        
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            # Count nodes
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            print(f"  Found {count} nodes")
            
            if count > 0:
                # Delete in batches for large graphs
                if count > 10000:
                    print("  Deleting in batches...")
                    while True:
                        result = session.run(
                            "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) as deleted"
                        )
                        deleted = result.single()["deleted"]
                        if deleted == 0:
                            break
                        print(f"    Deleted {deleted} nodes...")
                else:
                    session.run("MATCH (n) DETACH DELETE n")
                
                print(f"Neo4j: Cleared {count} nodes!")
            else:
                print("Neo4j: No nodes found")
        
        driver.close()
    except Exception as e:
        print(f"Neo4j error: {e}")


def main():
    print("=" * 50)
    print("CLEARING ALL DATABASES")
    print("=" * 50)
    
    clear_weaviate()
    clear_opensearch()
    clear_neo4j()
    
    print("\n" + "=" * 50)
    print("ALL DATABASES CLEARED!")
    print("=" * 50)


if __name__ == "__main__":
    main()
