"""Search Weaviate for summer term content."""
import weaviate
from weaviate.classes.query import Filter

client = weaviate.connect_to_local(port=8090)

try:
    collection = client.collections.get("VietnameseDocumentV3")
    
    # Check schema first
    print("=== Collection Schema ===")
    config = collection.config.get()
    print(f"Properties: {[p.name for p in config.properties]}")
    print(f"Vectorizer: {config.vectorizer}")
    print(f"Vector config: {config.vector_config}")
    
    # Check if vectors exist
    print("\n=== Checking vectors ===")
    sample = collection.query.fetch_objects(limit=3, include_vector=True)
    for obj in sample.objects:
        vec = obj.vector
        if vec:
            # Handle both dict and list formats
            if isinstance(vec, dict):
                vec_list = list(vec.values())[0] if vec else []
            else:
                vec_list = vec
            print(f"  Vector length: {len(vec_list) if vec_list else 0}")
        else:
            print(f"  NO VECTOR!")
    
    # Try vector search with embedding
    print("\n=== Vector Search (near_text) ===")
    try:
        results = collection.query.near_text(
            query="học kỳ hè đăng ký tín chỉ",
            limit=5,
            return_properties=["text", "title"]
        )
        print(f"Found {len(results.objects)} results")
        for obj in results.objects:
            print(f"  - {obj.properties.get('title', 'N/A')[:60]}")
    except Exception as e:
        print(f"Vector search error: {e}")

finally:
    client.close()
