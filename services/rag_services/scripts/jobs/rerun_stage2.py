"""
Re-run Stage 2 (LLM Semantic Extraction) on existing extraction JSON.
"""
import json
import sys
from pathlib import Path

# =============================================================================
# PATH SETUP - Critical for resolving app.core modules after refactoring
# =============================================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
SCRIPTS_DIR = SCRIPT_DIR.parent
RAG_SERVICES_ROOT = SCRIPTS_DIR.parent

# Add rag_services root to sys.path
if str(RAG_SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_SERVICES_ROOT))

# Import from new app.core location
from app.core.extraction.hybrid_extractor import SemanticExtractor, LLMConfig

def main():
    # Load existing extraction
    input_file = Path(__file__).parent.parent / "data" / "extraction_results" / "extraction_20251205_164642_1ec436cf.json"
    
    print(f"Loading: {input_file.name}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get articles from stage1
    articles = data['stage1_structure'].get('articles', [])
    print(f"Found {len(articles)} articles")

    # Initialize LLM extractor
    config = LLMConfig.from_env()
    print(f"Using model: {config.model}")
    extractor = SemanticExtractor(config)

    # Extract semantics
    all_entities = []
    all_relations = []
    errors = []

    for i, article in enumerate(articles):
        article_id = article.get('id', '')
        print(f"[{i+1}/{len(articles)}] Processing {article_id}...")
        try:
            result = extractor.extract_from_article(
                article_id=article_id,
                article_title=article.get('title', ''),
                article_text=article.get('full_text', '')
            )
            
            if hasattr(result, 'model_dump'):
                result_dict = result.model_dump()
            else:
                result_dict = result
            
            nodes = result_dict.get('nodes', [])
            edges = result_dict.get('edges', [])
            
            for node in nodes:
                node['source_article_id'] = article_id
                all_entities.append(node)
            
            for edge in edges:
                edge['source_article_id'] = article_id
                all_relations.append(edge)
            
            print(f"  -> {len(nodes)} entities, {len(edges)} relations")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            errors.append({'article_id': article_id, 'error': str(e)})

    print(f"\n{'='*50}")
    print(f"DONE")
    print(f"{'='*50}")
    print(f"Entities: {len(all_entities)}")
    print(f"Relations: {len(all_relations)}")
    print(f"Errors: {len(errors)}")

    # Update data
    data['stage2_semantic'] = {
        'entities': all_entities,
        'relations': all_relations,
        'errors': errors,
        'stats': {
            'entities': len(all_entities),
            'relations': len(all_relations),
            'errors': len(errors)
        }
    }

    # Save updated file
    output_path = input_file.parent / f"{input_file.stem}_stage2.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to: {output_path.name}")


if __name__ == "__main__":
    main()
