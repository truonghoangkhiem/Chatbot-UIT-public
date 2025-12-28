#!/usr/bin/env python3
"""Import merged JSON file to Neo4j"""
import json
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

from core.indexing.graph_builder import Neo4jGraphBuilder

# Load the merged file
json_path = 'data/extraction_results/merged_20251211_134623_3149bf44.json'
print(f"Loading {json_path}...")

with open(json_path, encoding='utf-8') as f:
    data = json.load(f)

print(f"Keys: {list(data.keys())}")

# Check stage1_structure for articles
stage1 = data.get('stage1_structure', {})
articles = stage1.get('articles', [])
print(f"Articles in stage1: {len(articles)}")

# Find English articles
eng_articles = [a for a in articles if 'tiếng anh' in a.get('title', '').lower() or 'tiếng anh' in a.get('full_text', '').lower()]
print(f"English articles: {len(eng_articles)}")
for a in eng_articles:
    print(f"  - {a.get('title')}")

# Import to Neo4j
print("\n" + "=" * 60)
print("Importing to Neo4j...")
print("=" * 60)

builder = Neo4jGraphBuilder()

# Don't clear - just add
result = builder.build_graph(data, clear_existing=False)
print(f"\nImport result: {result}")

# Check stats after import
stats = builder.get_graph_stats()
print(f"\nNeo4j stats after import:")
print(f"  Nodes: {stats.get('total_nodes')}")
print(f"  Relations: {stats.get('total_relations')}")
