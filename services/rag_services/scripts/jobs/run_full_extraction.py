"""
Run Full Two-Stage Pipeline Extraction and Save Results.

This script runs the complete extraction pipeline on the legal document PDF
and saves the results to a JSON file for analysis.

Author: Legal Document Processing Team
Date: 2024
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# =============================================================================
# PATH SETUP - Critical for resolving app.core modules after refactoring
# =============================================================================
# Get the rag_services root directory (2 levels up from scripts/jobs/)
SCRIPT_DIR = Path(__file__).parent.absolute()
SCRIPTS_DIR = SCRIPT_DIR.parent
RAG_SERVICES_ROOT = SCRIPTS_DIR.parent

# Add rag_services root to sys.path so we can import from app.core
if str(RAG_SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_SERVICES_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment
from dotenv import load_dotenv
load_dotenv(dotenv_path=RAG_SERVICES_ROOT / ".env")

import os
print(f"\n{'='*60}")
print("FULL EXTRACTION PIPELINE")
print(f"{'='*60}")

api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("ERROR: No API key found!")
    sys.exit(1)
print(f"API Key: ...{api_key[-8:]}")
print(f"Base URL: {os.getenv('OPENAI_BASE_URL')}")

# Import pipeline from new app.core location
from app.core.extraction.hybrid_extractor import (
    VLMConfig, VLMProvider, StructureExtractor,
    LLMConfig, SemanticExtractor,
    StructureExtractionResult, SemanticExtractionResult,
    run_pipeline
)


def run_structure_extraction_only(
    image_dir: Path,
    output_path: Path,
    max_pages: int = None
):
    """
    Run Stage 1 (Structure Extraction) only and save results.
    
    Args:
        image_dir: Directory containing page images
        output_path: Path to save JSON results
        max_pages: Max pages to process (None = all)
    """
    print(f"\n[STAGE 1] Structural Extraction")
    print(f"-" * 40)
    
    # Setup config
    config = VLMConfig.from_env(VLMProvider.OPENROUTER)
    config.model = "openai/gpt-4.1"  # Model đã test hoạt động
    print(f"VLM: {config.model}")
    
    # Find images
    images = sorted(image_dir.glob("*.png"))
    if max_pages:
        images = images[:max_pages]
    print(f"Processing {len(images)} pages")
    
    # Create extractor and run
    extractor = StructureExtractor(config)
    result = extractor.extract_from_images(images, continue_on_error=True)
    
    # Prepare output data
    output_data = {
        "extraction_time": datetime.now().isoformat(),
        "source": str(image_dir),
        "page_count": result.page_count,
        "stats": {
            "chapters": len(result.chapters),
            "articles": len(result.articles),
            "clauses": len(result.clauses),
            "relations": len(result.relations),
            "errors": len(result.errors)
        },
        "document": result.document.model_dump() if result.document else None,
        "chapters": [c.model_dump() for c in result.chapters],
        "articles": [a.model_dump() for a in result.articles],
        "clauses": [cl.model_dump() for cl in result.clauses],
        "relations": [r.model_dump() for r in result.relations],
        "errors": result.errors
    }
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("STAGE 1 COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  - Chapters: {len(result.chapters)}")
    print(f"  - Articles: {len(result.articles)}")
    print(f"  - Clauses: {len(result.clauses)}")
    print(f"  - Relations: {len(result.relations)}")
    
    if result.errors:
        print(f"\n  ⚠️ Errors: {len(result.errors)}")
        for err in result.errors:
            print(f"    - Page {err.get('page_number')}: {err.get('error')}")
    
    # Show cross-page nodes
    cross_page_articles = [a for a in result.articles if len(a.page_range) > 1]
    if cross_page_articles:
        print(f"\n  📄 Cross-page Articles: {len(cross_page_articles)}")
        for art in cross_page_articles:
            print(f"    - {art.id}: pages {art.page_range}, {len(art.full_text)} chars")
    
    return result, output_data


def run_semantic_extraction(
    structure_result: StructureExtractionResult,
    output_path: Path
):
    """
    Run Stage 2 (Semantic Extraction) using LLM.
    
    Args:
        structure_result: Result from Stage 1
        output_path: Path to save JSON results
    """
    print(f"\n[STAGE 2] Semantic Extraction with LLM")
    print(f"-" * 40)
    
    # Setup config
    config = LLMConfig.from_env()
    print(f"LLM: {config.model}")
    
    # Create extractor
    extractor = SemanticExtractor(config)
    
    all_entities = []
    all_relations = []
    all_errors = []
    
    # Process each article AND clause
    items_to_process = []
    
    # Add articles
    for article in structure_result.articles:
        if article.full_text.strip():
            items_to_process.append({
                "id": article.id,
                "title": article.title,
                "text": article.full_text,
                "type": "Article"
            })
    
    # Add clauses (they contain the detailed rules)
    for clause in structure_result.clauses:
        if clause.full_text.strip():
            items_to_process.append({
                "id": clause.id,
                "title": clause.title,
                "text": clause.full_text,
                "type": "Clause"
            })
    
    print(f"Processing {len(items_to_process)} items (articles + clauses)")
    
    for item in items_to_process:
        print(f"\n  Processing: {item['id']} ({item['type']})")
        
        try:
            result = extractor.extract_from_article(
                article_id=item["id"],
                article_title=item["title"],
                article_text=item["text"]
            )
            
            all_entities.extend([e.model_dump() for e in result.nodes])
            all_relations.extend([r.model_dump() for r in result.relations])
            
            if result.errors:
                all_errors.extend(result.errors)
            
            print(f"    -> {len(result.nodes)} entities, {len(result.relations)} relations")
            
        except Exception as e:
            print(f"    -> ERROR: {e}")
            all_errors.append({"item_id": item["id"], "error": str(e)})
    
    # Prepare output
    output_data = {
        "extraction_time": datetime.now().isoformat(),
        "stage": "semantic_extraction",
        "stats": {
            "items_processed": len(items_to_process),
            "entities": len(all_entities),
            "relations": len(all_relations),
            "errors": len(all_errors)
        },
        "entities": all_entities,
        "relations": all_relations,
        "errors": all_errors
    }
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("STAGE 2 COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  - Items processed: {len(items_to_process)}")
    print(f"  - Entities extracted: {len(all_entities)}")
    print(f"  - Relations extracted: {len(all_relations)}")
    
    if all_errors:
        print(f"  - Errors: {len(all_errors)}")
    
    return output_data


def convert_pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 200) -> list:
    """Convert PDF to PNG images."""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("ERROR: pdf2image not installed. Run: pip install pdf2image")
        return []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting PDF to images (DPI={dpi})...")
    images = convert_from_path(str(pdf_path), dpi=dpi, fmt="png")
    
    image_paths = []
    for i, image in enumerate(images, start=1):
        filename = f"page_{str(i).zfill(3)}.png"
        image_path = output_dir / filename
        image.save(str(image_path), "PNG")
        image_paths.append(image_path)
        print(f"  - Saved {filename}")
    
    print(f"Converted {len(images)} pages")
    return image_paths


def main():
    """Main entry point - runs both Stage 1 (VLM) and Stage 2 (LLM)."""
    # Define paths
    data_dir = Path(__file__).parent.parent / "data" / "quy_dinh"
    pdf_name = "1393-qd-dhcntt_29-12-2023_cap_nhat_quy_che_dao_tao_theo_hoc_che_tin_chi_cho_he_dai_hoc_chinh_quy"
    pdf_path = data_dir / f"{pdf_name}.pdf"
    image_dir = data_dir / f"{pdf_name}_images"
    
    output_dir = Path(__file__).parent.parent / "data" / "extraction_results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    structure_output = output_dir / f"stage1_structure_{timestamp}.json"
    semantic_output = output_dir / f"stage2_semantic_{timestamp}.json"
    combined_output = output_dir / f"full_extraction_{timestamp}.json"
    
    # Check if we need to convert PDF to images
    images = sorted(image_dir.glob("*.png")) if image_dir.exists() else []
    
    if not images:
        if not pdf_path.exists():
            print(f"ERROR: PDF not found: {pdf_path}")
            return
        
        print(f"\nNo images found. Converting PDF...")
        images = convert_pdf_to_images(pdf_path, image_dir)
        
        if not images:
            print("ERROR: Failed to convert PDF")
            return
    
    print(f"\nFound {len(images)} page images")
    
    # =========================================================================
    # STAGE 1: Structure Extraction (VLM)
    # =========================================================================
    structure_result, structure_data = run_structure_extraction_only(
        image_dir=image_dir,
        output_path=structure_output,
        max_pages=None
    )
    
    # =========================================================================
    # STAGE 2: Semantic Extraction (LLM)
    # =========================================================================
    semantic_data = run_semantic_extraction(
        structure_result=structure_result,
        output_path=semantic_output
    )
    
    # =========================================================================
    # COMBINE RESULTS
    # =========================================================================
    print(f"\n{'='*60}")
    print("COMBINING RESULTS")
    print(f"{'='*60}")
    
    combined_data = {
        "extraction_time": datetime.now().isoformat(),
        "source": str(pdf_path),
        "pipeline": "Two-Stage VLM+LLM",
        "stage1_structure": structure_data,
        "stage2_semantic": semantic_data,
        "summary": {
            "total_structural_nodes": (
                len(structure_data.get("chapters", [])) +
                len(structure_data.get("articles", [])) +
                len(structure_data.get("clauses", []))
            ),
            "total_structural_relations": len(structure_data.get("relations", [])),
            "total_semantic_entities": len(semantic_data.get("entities", [])),
            "total_semantic_relations": len(semantic_data.get("relations", []))
        }
    }
    
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Combined results saved to: {combined_output}")
    print(f"\n{'='*60}")
    print("FULL PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  1. Stage 1 (Structure): {structure_output.name}")
    print(f"  2. Stage 2 (Semantic):  {semantic_output.name}")
    print(f"  3. Combined:            {combined_output.name}")
    
    print(f"\nFinal Summary:")
    print(f"  📄 Structural: {combined_data['summary']['total_structural_nodes']} nodes, "
          f"{combined_data['summary']['total_structural_relations']} relations")
    print(f"  🔗 Semantic:   {combined_data['summary']['total_semantic_entities']} entities, "
          f"{combined_data['summary']['total_semantic_relations']} relations")


if __name__ == "__main__":
    main()
