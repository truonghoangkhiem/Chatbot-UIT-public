#!/usr/bin/env python3
"""Test Weaviate V3 data quality."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from infrastructure.store.vector.weaviate_store import get_weaviate_client
from weaviate.classes.query import Filter

client = get_weaviate_client("http://localhost:8090")
collection = client.collections.get("VietnameseDocumentV3")

print("\n" + "="*80)
print("🔍 WEAVIATE V3 DATA QUALITY TEST")
print("="*80)

# Count
agg = collection.aggregate.over_all()
print(f"\n📊 Total objects: {agg.total_count}")

# Sample
response = collection.query.fetch_objects(limit=3)
print(f"\n📄 Sample (first 3):")
for i, obj in enumerate(response.objects, 1):
    p = obj.properties
    print(f"  {i}. {p.get('article', 'N/A')} - {p.get('title', '')[:50]}")
    print(f"     Type: {p.get('structure_type')} | Chapter: {p.get('chapter')}")
    print(f"     Text: {p.get('text', '')[:80]}...")

# Check quality
response = collection.query.fetch_objects(limit=10)
flattened = sum(1 for o in response.objects if o.properties.get('structure_type'))
toc_count = sum(1 for o in response.objects if '......' in o.properties.get('text', ''))

print(f"\n✅ Quality Metrics:")
print(f"   Flattened metadata: {flattened}/10")
print(f"   TOC artifacts: {toc_count}/10")

# Test filters
articles = collection.query.fetch_objects(
    filters=Filter.by_property("structure_type").equal("article"), limit=5
)
print(f"   Articles found: {len(articles.objects)}")

client.close()
print("\n🎉 Weaviate V3 is working!\n")
