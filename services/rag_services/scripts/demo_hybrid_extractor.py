"""
Demo script for the Hybrid Two-Stage Knowledge Graph Extraction Pipeline.

This script demonstrates how to use the hybrid_extractor module to extract
knowledge graphs from legal document images or PDFs.

Usage:
    python demo_hybrid_extractor.py

Requirements:
    - OPENROUTER_API_KEY in .env file
    - VLM_MODEL (optional, defaults to google/gemini-flash-1.5)
    - LLM_MODEL (optional, defaults to openai/gpt-4o-mini)
"""

import sys
import json
from pathlib import Path

# Add parent directories to path for imports
RAG_SERVICES_ROOT = Path(__file__).parent.parent
if str(RAG_SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_SERVICES_ROOT))

# Import from new app.core location
from app.core.extraction.hybrid_extractor import (
    StructureExtractor,
    SemanticExtractor,
    VLMConfig,
    LLMConfig,
    VLMProvider,
    StructureNode,
    StructureNodeType,
    run_pipeline,
    convert_to_graph_models,
    HybridExtractionResult,
    StructureExtractionResult,
)


def demo_structure_extractor():
    """Demo Stage 1: Structure Extraction."""
    print("\n" + "=" * 60)
    print("DEMO: Stage 1 - Structure Extraction")
    print("=" * 60)
    
    # Create mock structure result (simulating VLM output)
    mock_articles = [
        StructureNode(
            id="dieu_1",
            type=StructureNodeType.ARTICLE,
            title="Điều 1. Cập nhật Quy chế Đào tạo",
            full_text="""Điều 1. Cập nhật Quy chế Đào tạo theo học chế tín chỉ được ban hành 
theo Quyết định số 790/QĐ-ĐHCNTT ngày 28 tháng 9 năm 2022 của Hiệu trưởng 
Trường Đại học Công nghệ Thông tin như sau:

Khoản 3 Điều 4: Đối với học kỳ hè: Học phí = HPTCHM HKHE × TCHPHM + HPTCHL × TCHPHL + HPTCCT × TCHPCT
Trong đó:
- Với sinh viên còn trong thời gian học theo thiết kế chương trình đào tạo: 
  HPTCHM HKHE là mức chênh lệch giữa HPTCHM học kỳ hè của khóa tuyển sinh gần nhất 
  và HPTCHM học kỳ chính của từng khóa được quy định trong quy định học phí hiện hành.
- Với sinh viên đã quá thời gian học theo thiết kế chương trình đào tạo: 
  HPTCHM HKHE thực hiện theo quy định học phí hiện hành.""",
            page_range=[1]
        ),
        StructureNode(
            id="dieu_12_khoan_1",
            type=StructureNodeType.ARTICLE,
            title="Khoản 1 Điều 12. Mở lớp học phần",
            full_text="""Khoản 1 Điều 12: Lớp học phần được mở nếu số lượng sinh viên đăng ký 
tối thiểu là 70 sinh viên đối với các môn học đại cương và các môn học 
cơ sở của ngành học chung toàn Trường.""",
            page_range=[2]
        ),
        StructureNode(
            id="dieu_14_khoan_1",
            type=StructureNodeType.ARTICLE,
            title="Mục b khoản 1 Điều 14. Đăng ký học kỳ hè",
            full_text="""Mục b khoản 1 Điều 14: Trong học kỳ hè
- Tổng số tín chỉ đăng ký không được vượt quá 12 tín chỉ.
- Sinh viên được đăng ký học mới, học lại và cải thiện điểm nếu có nhu cầu.
- Trường thực hiện mở các lớp đủ sĩ số theo quy định, không mở lớp sĩ số ít.""",
            page_range=[2]
        ),
    ]
    
    print(f"\nMock articles created: {len(mock_articles)}")
    for article in mock_articles:
        print(f"  - {article.id}: {article.title[:50]}...")
    
    return mock_articles


def demo_semantic_extractor(articles):
    """Demo Stage 2: Semantic Extraction."""
    print("\n" + "=" * 60)
    print("DEMO: Stage 2 - Semantic Extraction")
    print("=" * 60)
    
    # Note: This requires API key to be set
    try:
        llm_config = LLMConfig.from_env()
        print(f"Using LLM: {llm_config.model}")
        
        extractor = SemanticExtractor(llm_config)
        
        # Extract from first article as demo
        article = articles[0]
        print(f"\nExtracting from: {article.id}")
        
        result = extractor.extract_from_article(
            article_id=article.id,
            article_title=article.title,
            article_text=article.full_text
        )
        
        print(f"\nExtracted entities ({len(result.nodes)}):")
        for node in result.nodes[:5]:  # Show first 5
            print(f"  - [{node.type}] {node.text}")
        
        print(f"\nExtracted relations ({len(result.relations)}):")
        for rel in result.relations[:5]:  # Show first 5
            print(f"  - {rel.source_id} --{rel.type}--> {rel.target_id}")
        
        return result
        
    except ValueError as e:
        print(f"\nSkipping semantic extraction: {e}")
        print("Set OPENROUTER_API_KEY or OPENAI_API_KEY to enable.")
        return None


def demo_graph_conversion():
    """Demo conversion to graph_models.py types."""
    print("\n" + "=" * 60)
    print("DEMO: Conversion to GraphNode/GraphRelationship")
    print("=" * 60)
    
    # Create mock hybrid result
    from app.core.extraction.hybrid_extractor import (
        SemanticNode,
        SemanticRelation,
    )
    
    mock_semantic_nodes = [
        SemanticNode(
            id="hoc_phi_hk_he",
            type="HOC_PHI",
            text="học phí học kỳ hè",
            normalized="Học phí HK Hè",
            confidence=0.95,
            source_article_id="dieu_1"
        ),
        SemanticNode(
            id="sinh_vien_chinh_quy",
            type="SINH_VIEN",
            text="sinh viên chính quy",
            normalized="Sinh viên chính quy",
            confidence=0.9,
            source_article_id="dieu_1"
        ),
        SemanticNode(
            id="dieu_kien_70_sv",
            type="DIEU_KIEN",
            text="tối thiểu 70 sinh viên",
            normalized="Điều kiện mở lớp",
            confidence=0.85,
            source_article_id="dieu_12_khoan_1"
        ),
        SemanticNode(
            id="tin_chi_12",
            type="DIEU_KIEN",
            text="không vượt quá 12 tín chỉ",
            normalized="Giới hạn tín chỉ HK Hè",
            confidence=0.9,
            source_article_id="dieu_14_khoan_1"
        ),
    ]
    
    mock_semantic_relations = [
        SemanticRelation(
            source_id="hoc_phi_hk_he",
            target_id="sinh_vien_chinh_quy",
            type="AP_DUNG_CHO",
            confidence=0.9,
            evidence="Học phí áp dụng cho sinh viên chính quy",
            source_article_id="dieu_1"
        ),
        SemanticRelation(
            source_id="dieu_kien_70_sv",
            target_id="sinh_vien_chinh_quy",
            type="YEU_CAU",
            confidence=0.85,
            evidence="Lớp học phần mở khi đủ 70 sinh viên",
            source_article_id="dieu_12_khoan_1"
        ),
    ]
    
    mock_result = HybridExtractionResult(
        structure=StructureExtractionResult(
            articles=demo_structure_extractor(),
            page_count=2
        ),
        semantic_nodes=mock_semantic_nodes,
        semantic_relations=mock_semantic_relations,
        total_pages=2,
        total_articles_processed=3
    )
    
    # Convert to graph models
    graph_nodes, graph_rels = convert_to_graph_models(mock_result)
    
    print(f"\nConverted to {len(graph_nodes)} GraphNodes:")
    for node in graph_nodes[:5]:
        print(f"  - [{node.category.value}] {node.id}: {node.properties.get('text', node.properties.get('title', ''))[:40]}...")
    
    print(f"\nConverted to {len(graph_rels)} GraphRelationships:")
    for rel in graph_rels:
        print(f"  - {rel.source_id} --{rel.rel_type.value}--> {rel.target_id}")
    
    return graph_nodes, graph_rels


def demo_full_pipeline_mock():
    """Demo the full pipeline with mock data (no API calls)."""
    print("\n" + "=" * 60)
    print("DEMO: Full Pipeline (Mock Data)")
    print("=" * 60)
    
    # This demonstrates the pipeline flow without actual API calls
    print("\nPipeline flow:")
    print("  1. Input: Page images or PDF")
    print("  2. Stage 1: VLM extracts Document -> Chapter -> Article structure")
    print("  3. Stage 2: LLM extracts Entity + Relation from each Article")
    print("  4. Output: GraphNode + GraphRelationship (from graph_models.py)")
    
    # Run demo steps
    articles = demo_structure_extractor()
    demo_graph_conversion()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo Hybrid Extractor")
    parser.add_argument(
        "--with-api",
        action="store_true",
        help="Run demo with actual API calls (requires API key)"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="Process a real PDF file"
    )
    parser.add_argument(
        "--images",
        type=str,
        help="Directory containing page images"
    )
    
    args = parser.parse_args()
    
    if args.pdf or args.images:
        # Run actual pipeline
        print("Running actual pipeline...")
        
        try:
            if args.pdf:
                result, nodes, rels = run_pipeline(
                    pdf_path=Path(args.pdf),
                    output_path=Path("hybrid_result.json")
                )
            else:
                image_dir = Path(args.images)
                image_paths = sorted([
                    p for p in image_dir.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ])
                result, nodes, rels = run_pipeline(
                    image_paths=image_paths,
                    output_path=Path("hybrid_result.json")
                )
            
            print(f"\nResult: {len(nodes)} nodes, {len(rels)} relationships")
            
        except Exception as e:
            print(f"Error: {e}")
            return 1
    
    elif args.with_api:
        # Run demo with API calls
        articles = demo_structure_extractor()
        demo_semantic_extractor(articles)
        demo_graph_conversion()
    
    else:
        # Run mock demo (no API calls)
        demo_full_pipeline_mock()
    
    return 0


if __name__ == "__main__":
    exit(main())
