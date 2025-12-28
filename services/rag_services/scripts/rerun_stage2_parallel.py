"""
Re-run Stage 2 (Semantic Extraction) with Parallel Processing.

Usage:
    python rerun_stage2_parallel.py <input_json> [--concurrency 5]
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path
RAG_SERVICES_ROOT = Path(__file__).parent.parent
if str(RAG_SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_SERVICES_ROOT))

# Import from new app.core location
from app.core.extraction.hybrid_extractor import ParallelSemanticExtractor, LLMConfig


def main():
    parser = argparse.ArgumentParser(description="Re-run Stage 2 with parallel processing")
    parser.add_argument("input_file", help="Input JSON file from extraction")
    parser.add_argument("--concurrency", "-c", type=int, default=5, help="Max concurrent requests (default: 5)")
    parser.add_argument("--output", "-o", help="Output file (default: same file with _parallel suffix)")
    
    args = parser.parse_args()
    
    # Load input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return
    
    print(f"\n{'='*60}")
    print("PARALLEL SEMANTIC EXTRACTION (Stage 2)")
    print(f"{'='*60}")
    print(f"Input: {input_path.name}")
    print(f"Concurrency: {args.concurrency}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get articles from stage1
    stage1 = data.get("stage1_structure", {})
    articles = stage1.get("articles", [])
    
    print(f"Articles to process: {len(articles)}")
    
    if not articles:
        print("ERROR: No articles found in stage1_structure")
        return
    
    # Initialize parallel extractor
    config = LLMConfig.from_env()
    extractor = ParallelSemanticExtractor(
        config=config,
        max_concurrency=args.concurrency,
        max_retries=3,
        retry_delay=1.0
    )
    
    print(f"\n{'='*60}")
    print("STARTING PARALLEL EXTRACTION...")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    
    # Run parallel extraction
    results = extractor.extract_batch(articles, show_progress=True)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Aggregate results
    all_entities = []
    all_relations = []
    errors = []
    
    for result in results:
        # Convert Pydantic models to dict
        for node in result.nodes:
            entity_dict = node.model_dump() if hasattr(node, 'model_dump') else dict(node)
            entity_dict['source_article_id'] = result.article_id
            all_entities.append(entity_dict)
        
        for rel in result.relations:
            rel_dict = rel.model_dump() if hasattr(rel, 'model_dump') else dict(rel)
            rel_dict['source_article_id'] = result.article_id
            all_relations.append(rel_dict)
        
        if result.errors:
            errors.extend([{"article_id": result.article_id, "error": e} for e in result.errors])
    
    # Update stage2_semantic
    data["stage2_semantic"] = {
        "entities": all_entities,
        "relations": all_relations,
        "errors": errors,
        "stats": {
            "entities": len(all_entities),
            "relations": len(all_relations),
            "errors": len(errors),
            "processing_time_seconds": elapsed,
            "concurrency": args.concurrency
        }
    }
    
    # Save output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_parallel.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(articles):.2f}s per article)")
    print(f"  Articles: {len(articles)}")
    print(f"  Entities: {len(all_entities)}")
    print(f"  Relations: {len(all_relations)}")
    print(f"  Errors: {len(errors)}")
    print(f"\nSaved to: {output_path}")
    
    # Compare with sequential estimate
    sequential_estimate = len(articles) * 2.5  # ~2.5s per article sequentially
    speedup = sequential_estimate / elapsed if elapsed > 0 else 0
    print(f"\nEstimated speedup: {speedup:.1f}x vs sequential")


if __name__ == "__main__":
    main()
