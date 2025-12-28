"""
Test Neo4j Graph Builder with extraction data.

Usage:
    python test_graph_builder.py
    
Requires:
    - Neo4j running on localhost:7687
    - Environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

import sys
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os

def main():
    print("\n" + "="*60)
    print("TEST: Neo4j Graph Builder")
    print("="*60)
    
    # Check Neo4j connection params
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    print(f"Neo4j URI: {uri}")
    print(f"Neo4j User: {user}")
    print(f"Neo4j Password: {'*' * len(password)}")
    
    # Find latest extraction JSON
    data_dir = Path(__file__).parent.parent / "data" / "extraction_results"
    json_files = sorted(data_dir.glob("full_extraction_*.json"), reverse=True)
    
    if not json_files:
        print("\nERROR: No extraction JSON files found!")
        print(f"Run the extraction pipeline first, or check: {data_dir}")
        return
    
    json_file = json_files[0]
    print(f"\nUsing: {json_file.name}")
    
    # Load data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nExtraction Summary:")
    print(f"  Stage 1 - Documents: {data['stage1_structure']['stats'].get('articles', 0)} articles")
    print(f"  Stage 2 - Entities:  {data['stage2_semantic']['stats'].get('entities', 0)} entities")
    print(f"  Stage 2 - Relations: {data['stage2_semantic']['stats'].get('relations', 0)} relations")
    
    # Import and build from new location
    from app.core.indexing.graph_builder import Neo4jGraphBuilder
    
    print(f"\n{'='*60}")
    print("BUILDING GRAPH...")
    print("="*60)
    
    try:
        with Neo4jGraphBuilder(uri=uri, user=user, password=password) as builder:
            # Clear and build fresh
            stats = builder.build_graph(
                extraction_data=data,
                category="Quy chế Đào tạo",
                clear_first=True  # Start fresh for testing
            )
            
            print(f"\n{'='*60}")
            print("RESULTS")
            print("="*60)
            print(f"  ✅ Documents:  {stats.documents}")
            print(f"  ✅ Articles:   {stats.articles}")
            print(f"  ✅ Clauses:    {stats.clauses}")
            print(f"  ✅ Entities:   {stats.entities}")
            print(f"  ✅ Merged:     {stats.entities_merged} (dedup)")
            print(f"  ✅ Mentions:   {stats.mentions}")
            print(f"  ✅ Struct Rels: {stats.structural_relations}")
            print(f"  ✅ Semantic:   {stats.semantic_relations}")
            
            # Test queries
            print(f"\n{'='*60}")
            print("TEST QUERIES")
            print("="*60)
            
            # Test 1: Find "sinh viên" entity
            print("\n📍 Finding 'sinh viên' entities:")
            results = builder.get_entity_by_text("sinh viên")
            for r in results[:3]:
                print(f"   - {r.get('type')}: {r.get('text')} (mentions: {r.get('mention_count', 1)})")
            
            # Test 2: Get articles in category
            print("\n📍 Articles in 'Quy chế Đào tạo':")
            articles = builder.get_category_articles("Quy chế Đào tạo")
            for a in articles:
                print(f"   - {a['id']}: {a['title'][:50]}...")
            
            # Test 3: Get entities for an article
            if articles:
                article_id = articles[0]['id']
                print(f"\n📍 Entities in '{article_id}':")
                entities = builder.get_article_entities(article_id)
                for e in entities[:5]:
                    ent = e['entity']
                    print(f"   - [{ent.get('type')}] {ent.get('text')}")
            
            # Final stats
            print(f"\n{'='*60}")
            print("GRAPH STATISTICS")
            print("="*60)
            graph_stats = builder.get_graph_stats()
            for key, value in graph_stats.items():
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        if "Connection refused" in str(e):
            print("\n💡 Make sure Neo4j is running:")
            print("   docker run -d --name neo4j -p 7687:7687 -p 7474:7474 \\")
            print("     -e NEO4J_AUTH=neo4j/password neo4j:latest")


if __name__ == "__main__":
    main()
