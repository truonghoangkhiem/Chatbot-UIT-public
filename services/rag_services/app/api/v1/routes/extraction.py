"""
API Routes for Knowledge Graph Extraction Pipeline.

Provides endpoints for:
- Stage 1: Upload PDF and extract structure (VLM only)
- Stage 2: Upload Stage 1 JSON and extract semantics (LLM)
- Full pipeline: Combined VLM + LLM
- Get extraction status
- Download extraction results
"""

import os
import json
import asyncio
import uuid
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

# Load .env file for Neo4j credentials
from dotenv import load_dotenv
load_dotenv()

router = APIRouter(prefix="/extraction", tags=["Knowledge Graph Extraction"])

# Store extraction jobs status
extraction_jobs: Dict[str, Dict[str, Any]] = {}

# Paths
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent  # rag_services/
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
RESULTS_DIR = BASE_DIR / "data" / "extraction_results"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ExtractionRequest(BaseModel):
    """Request model for extraction."""
    category: str = "Quy chế Đào tạo"
    push_to_neo4j: bool = False


class ExtractionStatus(BaseModel):
    """Status of an extraction job."""
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0-100
    current_step: str
    stage: str = "full"  # stage1, stage2, full
    created_at: str
    completed_at: Optional[str] = None
    result_file: Optional[str] = None
    error: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


class Stage2Request(BaseModel):
    """Request model for Stage 2 extraction."""
    stage1_data: Dict[str, Any]
    category: str = "Quy chế Đào tạo"
    push_to_neo4j: bool = False


def get_extraction_status(job_id: str) -> Optional[ExtractionStatus]:
    """Get status of an extraction job."""
    if job_id not in extraction_jobs:
        return None
    return ExtractionStatus(**extraction_jobs[job_id])


# =============================================================================
# Stage 1: VLM Structure Extraction Only
# =============================================================================

async def run_stage1_pipeline(
    job_id: str,
    pdf_path: Path,
    category: str
):
    """
    Run Stage 1: VLM for structural extraction only.
    
    Output: JSON with document structure (chapters, articles, clauses)
    """
    try:
        extraction_jobs[job_id]["status"] = "processing"
        extraction_jobs[job_id]["current_step"] = "Đang chuyển PDF sang ảnh..."
        extraction_jobs[job_id]["progress"] = 10
        
        # Import extraction modules
        from app.core.extraction.hybrid_extractor import StructureExtractor, VLMConfig
        from pdf2image import convert_from_path
        
        # Convert PDF to images
        images_dir = pdf_path.parent / f"{pdf_path.stem}_images"
        images_dir.mkdir(exist_ok=True)
        
        loop = asyncio.get_event_loop()
        pages = await loop.run_in_executor(
            None,
            lambda: convert_from_path(str(pdf_path), dpi=200)
        )
        
        image_paths = []
        for i, page in enumerate(pages):
            img_path = images_dir / f"page_{i+1}.png"
            await loop.run_in_executor(None, lambda p=page, ip=img_path: p.save(str(ip), "PNG"))
            image_paths.append(str(img_path))
        
        extraction_jobs[job_id]["progress"] = 20
        extraction_jobs[job_id]["current_step"] = f"Đã chuyển {len(pages)} trang. Đang khởi tạo VLM..."
        
        # Initialize VLM extractor
        vlm_config = VLMConfig.from_env()
        structure_extractor = StructureExtractor(vlm_config)
        
        # Stage 1: VLM Structure Extraction
        extraction_jobs[job_id]["current_step"] = "Stage 1: Trích xuất cấu trúc với VLM..."
        extraction_jobs[job_id]["progress"] = 30
        
        structure_result = await loop.run_in_executor(
            None,
            lambda: structure_extractor.extract_from_images(image_paths)
        )
        
        # Convert Pydantic model to dict
        if hasattr(structure_result, 'model_dump'):
            structure_dict = structure_result.model_dump()
        else:
            structure_dict = structure_result
        
        # POST-PROCESSING: Auto-fix relations for amendment documents
        from app.core.extraction.page_merger import auto_fix_amendment_relations
        structure_dict = auto_fix_amendment_relations(structure_dict)
        
        # POST-PROCESSING: Clean and validate extraction result
        from app.core.extraction.cleaner import clean_extraction_result
        
        extraction_jobs[job_id]["current_step"] = "Đang dọn dẹp và xác thực kết quả..."
        extraction_jobs[job_id]["progress"] = 85
        
        # Wrap structure in result format for cleaner
        temp_result = {
            "source_file": pdf_path.name,
            "structure": structure_dict
        }
        cleaned_result, cleaning_stats = clean_extraction_result(temp_result)
        structure_dict = cleaned_result.get("structure", structure_dict)
        
        # Log cleaning stats
        if cleaning_stats.duplicate_nodes_removed > 0:
            logger.info(f"Removed {cleaning_stats.duplicate_nodes_removed} duplicate nodes")
        if cleaning_stats.invalid_modifications_removed > 0:
            logger.info(f"Removed {cleaning_stats.invalid_modifications_removed} invalid modifications")
        if cleaning_stats.is_original_document:
            logger.info("Document detected as ORIGINAL (not amendment) - modifications cleared")
        if cleaning_stats.errors:
            logger.warning(f"Cleaning detected {len(cleaning_stats.errors)} issues")
        
        extraction_jobs[job_id]["progress"] = 90
        extraction_jobs[job_id]["current_step"] = f"Stage 1 hoàn thành: {len(structure_dict.get('articles', []))} điều"
        
        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = {
            "extraction_id": job_id,
            "stage": "stage1",
            "source_file": pdf_path.name,
            "extracted_at": datetime.now().isoformat(),
            "category": category,
            "page_count": len(pages),
            "structure": structure_dict,
            "cleaning_applied": True,
            "document_type": "original" if cleaning_stats.is_original_document else "amendment",
            "cleaning_stats": {
                "duplicates_removed": cleaning_stats.duplicate_nodes_removed,
                "orphan_relations_removed": cleaning_stats.orphan_relations_removed,
                "invalid_modifications_removed": cleaning_stats.invalid_modifications_removed,
                "is_original_document": cleaning_stats.is_original_document,
                "issues_detected": len(cleaning_stats.errors),
                "issues": cleaning_stats.errors[:10] if cleaning_stats.errors else []
            }
        }
        
        result_file = RESULTS_DIR / f"stage1_{timestamp}_{job_id[:8]}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Complete
        extraction_jobs[job_id]["status"] = "completed"
        extraction_jobs[job_id]["progress"] = 100
        extraction_jobs[job_id]["current_step"] = "Stage 1 hoàn thành!"
        extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        extraction_jobs[job_id]["result_file"] = str(result_file.name)
        extraction_jobs[job_id]["stats"] = {
            "pages": len(pages),
            "chapters": len(structure_dict.get("chapters", [])),
            "articles": len(structure_dict.get("articles", [])),
            "clauses": len(structure_dict.get("clauses", [])),
            "duplicates_removed": cleaning_stats.duplicate_nodes_removed,
            "issues_detected": len(cleaning_stats.errors)
        }
        
        # Cleanup images
        shutil.rmtree(images_dir, ignore_errors=True)
        
    except Exception as e:
        import traceback
        extraction_jobs[job_id]["status"] = "failed"
        extraction_jobs[job_id]["error"] = str(e)
        extraction_jobs[job_id]["current_step"] = f"Lỗi: {str(e)}"
        print(f"Stage 1 extraction error: {traceback.format_exc()}")


# =============================================================================
# Stage 2: LLM Semantic Extraction (FIXED TABLE MERGE)
# =============================================================================

async def run_stage2_pipeline(
    job_id: str,
    stage1_data: Dict[str, Any],
    category: str,
    push_to_neo4j: bool
):
    """
    Run Stage 2: LLM for semantic extraction from Stage 1 results.
    Input: Stage 1 JSON (structure)
    Output: Combined JSON with structure + semantics
    """
    try:
        extraction_jobs[job_id]["status"] = "processing"
        extraction_jobs[job_id]["current_step"] = "Đang khởi tạo LLM..."
        extraction_jobs[job_id]["progress"] = 10
        
        # Import extraction modules
        from app.core.extraction.hybrid_extractor import SemanticExtractor, LLMConfig
        
        loop = asyncio.get_event_loop()
        
        # Initialize LLM extractor
        llm_config = LLMConfig.from_env()
        semantic_extractor = SemanticExtractor(llm_config)
        
        # 1. Parse Input & Validate
        if "stage1_structure" in stage1_data:
            structure = stage1_data["stage1_structure"]
        elif "structure" in stage1_data:
            structure = stage1_data["structure"]
        else:
            structure = stage1_data

        articles = structure.get("articles", [])
        clauses = structure.get("clauses", [])
        tables = structure.get("tables", [])
        relations = structure.get("relations", [])
        
        if not articles:
            raise ValueError(f"Không tìm thấy điều khoản trong dữ liệu Stage 1. Keys: {list(stage1_data.keys())}")
        
        # 2. CRITICAL: MERGE CLAUSES AND TABLES INTO ARTICLES LOGIC
        # ---------------------------------------------------------------------
        extraction_jobs[job_id]["current_step"] = f"Đang gộp {len(clauses)} khoản và {len(tables)} bảng vào văn bản..."
        
        # Map article by ID
        article_map = {a['id']: a for a in articles}
        
        # Map Child -> Parent from relations
        child_to_parent = {}
        for rel in relations:
            if rel.get("type") == "CONTAINS":
                child_to_parent[rel["target"]] = rel["source"]
        
        # MERGE CLAUSES INTO ARTICLES
        clause_merged_count = 0
        for clause in clauses:
            parent_id = child_to_parent.get(clause["id"])
            if parent_id and parent_id in article_map:
                parent_article = article_map[parent_id]
                # Append Clause content to Article Text
                clause_title = clause.get('title', '')
                clause_text = clause.get('full_text', '')
                append_text = f"\n\n--- {clause_title} ---\n{clause_text}\n"
                parent_article["full_text"] += append_text
                clause_merged_count += 1
        
        print(f"Job {job_id}: Successfully merged {clause_merged_count} clauses into articles.")
        
        # MERGE TABLES INTO ARTICLES
        table_merged_count = 0
        for table in tables:
            parent_id = child_to_parent.get(table["id"])
            if parent_id and parent_id in article_map:
                parent_article = article_map[parent_id]
                # Append Markdown Table to Article Text
                append_text = f"\n\n=== BẢNG THAM CHIẾU ({table.get('title', 'Bảng')}) ===\n{table.get('full_text', '')}\n========================\n"
                parent_article["full_text"] += append_text
                table_merged_count += 1
        
        print(f"Job {job_id}: Successfully merged {table_merged_count} tables into articles.")
        # ---------------------------------------------------------------------

        extraction_jobs[job_id]["progress"] = 20
        extraction_jobs[job_id]["current_step"] = f"Đang xử lý {len(articles)} điều..."
        
        # Process articles in parallel
        MAX_CONCURRENT = 5
        all_entities = []
        all_relations = []
        all_modifications = []
        errors = []
        
        async def process_article(article):
            try:
                article_result = await loop.run_in_executor(
                    None,
                    lambda: semantic_extractor.extract_from_article(
                        article_id=article.get('id', ''),
                        article_title=article.get('title', ''),
                        article_text=article.get('full_text', '') # Text đã có bảng
                    )
                )
                
                if hasattr(article_result, 'model_dump'):
                    article_dict = article_result.model_dump()
                else:
                    article_dict = article_result if isinstance(article_result, dict) else {}
                
                entities = []
                for entity in article_dict.get('nodes', []):
                    entity['source_article_id'] = article.get('id', '')
                    entities.append(entity)
                
                relations = []
                for rel in article_dict.get('relations', []):
                    rel['source_article_id'] = article.get('id', '')
                    relations.append(rel)
                
                # Collect modifications (for amendment documents)
                modifications = article_dict.get('modifications', [])
                
                return {"entities": entities, "relations": relations, "modifications": modifications, "error": None}
                
            except Exception as e:
                return {"entities": [], "relations": [], "modifications": [], "error": {"article_id": article.get('id', ''), "error": str(e)}}
        
        # Process in batches
        for batch_start in range(0, len(articles), MAX_CONCURRENT):
            batch = articles[batch_start:batch_start + MAX_CONCURRENT]
            batch_tasks = [process_article(article) for article in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            
            for result in batch_results:
                all_entities.extend(result["entities"])
                all_relations.extend(result["relations"])
                all_modifications.extend(result.get("modifications", []))
                if result["error"]:
                    errors.append(result["error"])
            
            processed = min(batch_start + MAX_CONCURRENT, len(articles))
            progress = 20 + int(60 * processed / len(articles)) if articles else 80
            extraction_jobs[job_id]["progress"] = progress
            extraction_jobs[job_id]["current_step"] = f"Stage 2: Đã xử lý {processed}/{len(articles)} điều..."
        
        semantic_result = {
            "entities": all_entities,
            "relations": all_relations,
            "modifications": all_modifications,
            "errors": errors,
            "stats": {
                "entities": len(all_entities),
                "relations": len(all_relations),
                "modifications": len(all_modifications),
                "errors": len(errors)
            }
        }
        
        extraction_jobs[job_id]["progress"] = 85
        extraction_jobs[job_id]["current_step"] = "Đang tổng hợp kết quả..."
        
        # Combine results (merge stage1 + stage2)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged_result = {
            "extraction_id": job_id,
            "stage": "merged",
            "source_file": stage1_data.get("source_file", "unknown"),
            "extracted_at": datetime.now().isoformat(),
            "category": category,
            "stage1_structure": structure, # Cấu trúc đã được update text (có bảng)
            "stage2_semantic": semantic_result
        }
        
        result_file = RESULTS_DIR / f"merged_{timestamp}_{job_id[:8]}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(merged_result, f, ensure_ascii=False, indent=2)
        
        extraction_jobs[job_id]["progress"] = 90
        
        # Push to Neo4j if requested
        if push_to_neo4j:
            extraction_jobs[job_id]["current_step"] = "Đang đẩy dữ liệu lên Neo4j..."
            try:
                from app.core.indexing.graph_builder import Neo4jGraphBuilder
                
                uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
                user = os.getenv("NEO4J_USER", "neo4j")
                password = os.getenv("NEO4J_PASSWORD", "password")
                
                with Neo4jGraphBuilder(uri=uri, user=user, password=password) as builder:
                    stats = builder.build_graph(
                        extraction_data=merged_result,
                        category=category,
                        clear_first=False
                    )
                    extraction_jobs[job_id]["stats"]["neo4j"] = {
                        "entities": stats.entities,
                        "merged": stats.entities_merged,
                        "relations": stats.semantic_relations
                    }
            except Exception as e:
                extraction_jobs[job_id]["stats"]["neo4j_error"] = str(e)
        
        # Complete
        extraction_jobs[job_id]["status"] = "completed"
        extraction_jobs[job_id]["progress"] = 100
        extraction_jobs[job_id]["current_step"] = "Stage 2 + Merge hoàn thành!"
        extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        extraction_jobs[job_id]["result_file"] = str(result_file.name)
        extraction_jobs[job_id]["stats"] = {
            "articles_processed": len(articles),
            "clauses_merged": clause_merged_count,
            "tables_merged": table_merged_count,
            "entities": len(all_entities),
            "relations": len(all_relations),
            "errors": len(errors)
        }
        
    except Exception as e:
        import traceback
        extraction_jobs[job_id]["status"] = "failed"
        extraction_jobs[job_id]["error"] = str(e)
        extraction_jobs[job_id]["current_step"] = f"Lỗi: {str(e)}"
        print(f"Stage 2 extraction error: {traceback.format_exc()}")


# =============================================================================
# Full Pipeline (Original - FIXED)
# =============================================================================

async def run_extraction_pipeline(
    job_id: str,
    pdf_path: Path,
    category: str,
    push_to_neo4j: bool
):
    """
    Run the Two-Stage extraction pipeline asynchronously.
    Stage 1: VLM Structure -> Merge Table -> Stage 2: Semantic
    """
    try:
        # Update status
        extraction_jobs[job_id]["status"] = "processing"
        extraction_jobs[job_id]["current_step"] = "Đang chuyển PDF sang ảnh..."
        extraction_jobs[job_id]["progress"] = 10
        
        # Import extraction modules
        from app.core.extraction.hybrid_extractor import StructureExtractor, SemanticExtractor, VLMConfig, LLMConfig
        from pdf2image import convert_from_path
        
        # Convert PDF to images
        images_dir = pdf_path.parent / f"{pdf_path.stem}_images"
        images_dir.mkdir(exist_ok=True)
        
        extraction_jobs[job_id]["current_step"] = "Đang chuyển PDF sang ảnh..."
        
        # Run in executor to not block
        loop = asyncio.get_event_loop()
        pages = await loop.run_in_executor(
            None,
            lambda: convert_from_path(str(pdf_path), dpi=200)
        )
        
        image_paths = []
        for i, page in enumerate(pages):
            img_path = images_dir / f"page_{i+1}.png"
            await loop.run_in_executor(None, lambda p=page, ip=img_path: p.save(str(ip), "PNG"))
            image_paths.append(str(img_path))
        
        extraction_jobs[job_id]["progress"] = 20
        extraction_jobs[job_id]["current_step"] = f"Đã chuyển {len(pages)} trang. Đang khởi tạo extractor..."
        
        # Initialize extractors
        vlm_config = VLMConfig.from_env()
        llm_config = LLMConfig.from_env()
        structure_extractor = StructureExtractor(vlm_config)
        semantic_extractor = SemanticExtractor(llm_config)
        
        # Stage 1: VLM Structure Extraction
        extraction_jobs[job_id]["current_step"] = "Stage 1: Trích xuất cấu trúc với VLM..."
        extraction_jobs[job_id]["progress"] = 30
        
        structure_result = await loop.run_in_executor(
            None,
            lambda: structure_extractor.extract_from_images(image_paths)
        )
        
        # Convert Pydantic model to dict
        if hasattr(structure_result, 'model_dump'):
            structure_dict = structure_result.model_dump()
        else:
            structure_dict = structure_result
        
        # POST-PROCESSING: Auto-fix relations for amendment documents
        from app.core.extraction.page_merger import auto_fix_amendment_relations
        structure_dict = auto_fix_amendment_relations(structure_dict)
        
        extraction_jobs[job_id]["progress"] = 50
        
        # ---------------------------------------------------------------------
        # CRITICAL: MERGE CLAUSES AND TABLES INTO ARTICLES (Fixes missing data in Stage 2)
        # ---------------------------------------------------------------------
        extraction_jobs[job_id]["current_step"] = "Đang gộp khoản và bảng vào văn bản..."
        
        articles = structure_dict.get('articles', [])
        clauses = structure_dict.get('clauses', [])
        tables = structure_dict.get('tables', [])
        relations = structure_dict.get('relations', [])
        
        article_map = {a['id']: a for a in articles}
        child_to_parent = {r["target"]: r["source"] for r in relations if r["type"] == "CONTAINS"}
        
        # MERGE CLAUSES INTO ARTICLES
        clause_merged = 0
        for clause in clauses:
            parent_id = child_to_parent.get(clause['id'])
            if parent_id and parent_id in article_map:
                clause_title = clause.get('title', '')
                clause_text = clause.get('full_text', '')
                article_map[parent_id]['full_text'] += f"\n\n--- {clause_title} ---\n{clause_text}\n"
                clause_merged += 1
        
        # MERGE TABLES INTO ARTICLES
        table_merged = 0
        for table in tables:
            parent_id = child_to_parent.get(table['id'])
            if parent_id and parent_id in article_map:
                # Update article text with table content
                article_map[parent_id]['full_text'] += f"\n\n=== BẢNG ({table.get('title')}) ===\n{table.get('full_text')}\n"
                table_merged += 1
        
        print(f"Full pipeline: Merged {clause_merged} clauses, {table_merged} tables.")
        # ---------------------------------------------------------------------
        
        # Stage 2: LLM Semantic Extraction
        extraction_jobs[job_id]["current_step"] = "Stage 2: Trích xuất ngữ nghĩa với LLM (song song)..."
        extraction_jobs[job_id]["progress"] = 60
        
        async def process_article(article):
            """Process a single article."""
            try:
                article_result = await loop.run_in_executor(
                    None,
                    lambda: semantic_extractor.extract_from_article(
                        article_id=article.get('id', ''),
                        article_title=article.get('title', ''),
                        article_text=article.get('full_text', '')
                    )
                )
                
                if hasattr(article_result, 'model_dump'):
                    article_dict = article_result.model_dump()
                else:
                    article_dict = article_result if isinstance(article_result, dict) else {}
                
                entities = []
                for entity in article_dict.get('nodes', []):
                    entity['source_article_id'] = article.get('id', '')
                    entities.append(entity)
                
                relations = []
                for rel in article_dict.get('relations', []):
                    rel['source_article_id'] = article.get('id', '')
                    relations.append(rel)
                
                # Collect modifications (for amendment documents)
                modifications = article_dict.get('modifications', [])
                
                return {"entities": entities, "relations": relations, "modifications": modifications, "error": None}
                
            except Exception as e:
                return {"entities": [], "relations": [], "modifications": [], "error": {"article_id": article.get('id', ''), "error": str(e)}}
        
        # Run all articles in parallel
        MAX_CONCURRENT = 5
        all_entities = []
        all_relations = []
        all_modifications = []
        errors = []
        
        for batch_start in range(0, len(articles), MAX_CONCURRENT):
            batch = articles[batch_start:batch_start + MAX_CONCURRENT]
            batch_tasks = [process_article(article) for article in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            
            for result in batch_results:
                all_entities.extend(result["entities"])
                all_relations.extend(result["relations"])
                all_modifications.extend(result.get("modifications", []))
                if result["error"]:
                    errors.append(result["error"])
            
            processed = min(batch_start + MAX_CONCURRENT, len(articles))
            progress = 60 + int(20 * processed / len(articles)) if articles else 80
            extraction_jobs[job_id]["progress"] = progress
            extraction_jobs[job_id]["current_step"] = f"Stage 2: Đã xử lý {processed}/{len(articles)} điều..."
        
        semantic_result = {
            "entities": all_entities,
            "relations": all_relations,
            "modifications": all_modifications,
            "errors": errors,
            "stats": {
                "entities": len(all_entities),
                "relations": len(all_relations),
                "modifications": len(all_modifications),
                "errors": len(errors)
            }
        }
        
        extraction_jobs[job_id]["progress"] = 80
        
        # Combine results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = {
            "extraction_id": job_id,
            "source_file": pdf_path.name,
            "extracted_at": datetime.now().isoformat(),
            "category": category,
            "stage1_structure": structure_dict,
            "stage2_semantic": semantic_result
        }
        
        # Save result
        result_file = RESULTS_DIR / f"extraction_{timestamp}_{job_id[:8]}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        extraction_jobs[job_id]["progress"] = 90
        
        # Push to Neo4j if requested
        if push_to_neo4j:
            extraction_jobs[job_id]["current_step"] = "Đang đẩy dữ liệu lên Neo4j..."
            try:
                from app.core.indexing.graph_builder import Neo4jGraphBuilder
                
                uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
                user = os.getenv("NEO4J_USER", "neo4j")
                password = os.getenv("NEO4J_PASSWORD", "password")
                
                with Neo4jGraphBuilder(uri=uri, user=user, password=password) as builder:
                    stats = builder.build_graph(
                        extraction_data=result,
                        category=category,
                        clear_first=False
                    )
                    extraction_jobs[job_id]["stats"]["neo4j"] = {
                        "entities": stats.entities,
                        "merged": stats.entities_merged,
                        "relations": stats.semantic_relations
                    }
            except Exception as e:
                extraction_jobs[job_id]["stats"]["neo4j_error"] = str(e)
        
        # Complete
        extraction_jobs[job_id]["status"] = "completed"
        extraction_jobs[job_id]["progress"] = 100
        extraction_jobs[job_id]["current_step"] = "Hoàn thành!"
        extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        extraction_jobs[job_id]["result_file"] = str(result_file.name)
        extraction_jobs[job_id]["stats"] = {
            "pages": len(pages),
            "articles": len(structure_dict.get("articles", [])),
            "clauses_merged": clause_merged,
            "tables_merged": table_merged,
            "entities": semantic_result.get("stats", {}).get("entities", 0),
            "relations": semantic_result.get("stats", {}).get("relations", 0)
        }
        
        # Cleanup images
        shutil.rmtree(images_dir, ignore_errors=True)
        
    except Exception as e:
        import traceback
        extraction_jobs[job_id]["status"] = "failed"
        extraction_jobs[job_id]["error"] = str(e)
        extraction_jobs[job_id]["current_step"] = f"Lỗi: {str(e)}"
        print(f"Extraction error: {traceback.format_exc()}")


@router.post("/upload", response_model=ExtractionStatus)
async def upload_and_extract(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = "Quy chế Đào tạo",
    push_to_neo4j: bool = False
):
    """
    Upload a PDF file and start knowledge graph extraction.
    
    Returns a job ID to track progress.
    """
    # Validate file
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file PDF")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    pdf_path = UPLOAD_DIR / safe_filename
    
    try:
        with open(pdf_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lưu file: {str(e)}")
    
    # Initialize job status
    extraction_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "current_step": "Đang khởi tạo...",
        "stage": "full",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result_file": None,
        "error": None,
        "stats": None
    }
    
    # Start extraction in background
    background_tasks.add_task(
        run_extraction_pipeline,
        job_id,
        pdf_path,
        category,
        push_to_neo4j
    )
    
    return ExtractionStatus(**extraction_jobs[job_id])


# =============================================================================
# Stage 1 Endpoint: VLM Only
# =============================================================================

@router.post("/stage1/upload", response_model=ExtractionStatus)
async def upload_and_extract_stage1(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = "Quy chế Đào tạo"
):
    """
    Stage 1: Upload PDF and extract document structure using VLM.
    
    Returns JSON with chapters, articles, clauses structure.
    This can then be used as input for Stage 2.
    """
    # Validate file
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file PDF")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = f"{timestamp}_{file.filename}"
    pdf_path = UPLOAD_DIR / safe_filename
    
    try:
        with open(pdf_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lưu file: {str(e)}")
    
    # Initialize job status
    extraction_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "current_step": "Đang khởi tạo Stage 1...",
        "stage": "stage1",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result_file": None,
        "error": None,
        "stats": None
    }
    
    # Start Stage 1 extraction in background
    background_tasks.add_task(
        run_stage1_pipeline,
        job_id,
        pdf_path,
        category
    )
    
    return ExtractionStatus(**extraction_jobs[job_id])


# =============================================================================
# Stage 2 Endpoint: LLM + Merge
# =============================================================================

@router.post("/stage2/process", response_model=ExtractionStatus)
async def process_stage2(
    background_tasks: BackgroundTasks,
    request: Stage2Request
):
    """
    Stage 2: Process Stage 1 JSON to extract semantic entities and relations.
    
    Input: JSON from Stage 1 (structure)
    Output: Merged JSON with structure + semantics
    """
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    extraction_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "current_step": "Đang khởi tạo Stage 2...",
        "stage": "stage2",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result_file": None,
        "error": None,
        "stats": None
    }
    
    # Start Stage 2 extraction in background
    background_tasks.add_task(
        run_stage2_pipeline,
        job_id,
        request.stage1_data,
        request.category,
        request.push_to_neo4j
    )
    
    return ExtractionStatus(**extraction_jobs[job_id])


@router.post("/stage2/upload", response_model=ExtractionStatus)
async def upload_stage1_json_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str = "Quy chế Đào tạo",
    push_to_neo4j: bool = False
):
    """
    Stage 2: Upload Stage 1 JSON file and process with LLM.
    
    Alternative to /stage2/process - accepts file upload instead of JSON body.
    """
    # Validate file
    if not file.filename.lower().endswith('.json'):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file JSON")
    
    try:
        content = await file.read()
        stage1_data = json.loads(content.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"File JSON không hợp lệ: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc file: {str(e)}")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    extraction_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "current_step": "Đang khởi tạo Stage 2...",
        "stage": "stage2",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result_file": None,
        "error": None,
        "stats": None
    }
    
    # Start Stage 2 extraction in background
    background_tasks.add_task(
        run_stage2_pipeline,
        job_id,
        stage1_data,
        category,
        push_to_neo4j
    )
    
    return ExtractionStatus(**extraction_jobs[job_id])


@router.get("/status/{job_id}", response_model=ExtractionStatus)
async def get_status(job_id: str):
    """Get the status of an extraction job."""
    status = get_extraction_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job không tồn tại")
    return status


@router.get("/result/{job_id}")
async def get_result(job_id: str):
    """Get the extraction result JSON."""
    status = get_extraction_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job không tồn tại")
    
    if status.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job chưa hoàn thành: {status.status}")
    
    result_path = RESULTS_DIR / status.result_file
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="File kết quả không tồn tại")
    
    with open(result_path, 'r', encoding='utf-8') as f:
        return json.load(f)


@router.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download the extraction result as JSON file."""
    status = get_extraction_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job không tồn tại")
    
    if status.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job chưa hoàn thành: {status.status}")
    
    result_path = RESULTS_DIR / status.result_file
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="File kết quả không tồn tại")
    
    return FileResponse(
        path=result_path,
        filename=status.result_file,
        media_type="application/json"
    )


@router.get("/jobs")
async def list_jobs():
    """List all extraction jobs."""
    return [ExtractionStatus(**job) for job in extraction_jobs.values()]


# =============================================================================
# Neo4j Import: Upload JSON and Push to Neo4j
# =============================================================================

class Neo4jImportRequest(BaseModel):
    """Request model for direct Neo4j import."""
    data: Dict[str, Any]


@router.post("/neo4j/upload")
async def upload_json_to_neo4j(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    clear_existing: bool = False
):
    """
    Upload a JSON file and push to Neo4j.
    
    Accepts extraction result JSON from Stage 1, Stage 2, or merged results.
    """
    # Validate file
    if not file.filename.lower().endswith('.json'):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file JSON")
    
    try:
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"File JSON không hợp lệ: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi đọc file: {str(e)}")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    extraction_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "current_step": "Đang chuẩn bị import Neo4j...",
        "stage": "neo4j_import",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result_file": None,
        "error": None,
        "stats": None
    }
    
    # Start import in background
    background_tasks.add_task(
        run_neo4j_import,
        job_id,
        data,
        file.filename,
        clear_existing
    )
    
    return ExtractionStatus(**extraction_jobs[job_id])


async def run_neo4j_import(
    job_id: str,
    data: Dict[str, Any],
    filename: str,
    clear_existing: bool
):
    """
    Import extraction data to Neo4j.
    """
    try:
        extraction_jobs[job_id]["status"] = "processing"
        extraction_jobs[job_id]["current_step"] = "Đang kết nối Neo4j..."
        extraction_jobs[job_id]["progress"] = 10
        
        from app.core.indexing.graph_builder import Neo4jGraphBuilder
        
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        extraction_jobs[job_id]["progress"] = 20
        extraction_jobs[job_id]["current_step"] = "Đang xử lý dữ liệu..."
        
        # Get category from data
        category = data.get("category", "Quy chế Đào tạo")
        
        with Neo4jGraphBuilder(uri=uri, user=user, password=password) as builder:
            extraction_jobs[job_id]["progress"] = 30
            extraction_jobs[job_id]["current_step"] = "Đang import vào Neo4j..."
            
            # Use build_graph method which handles all the logic
            stats = builder.build_graph(
                extraction_data=data,
                category=category,
                clear_first=clear_existing
            )
            
            extraction_jobs[job_id]["progress"] = 90
            
            # Get final stats
            db_stats = builder.get_graph_stats()
        
        extraction_jobs[job_id]["progress"] = 100
        extraction_jobs[job_id]["status"] = "completed"
        extraction_jobs[job_id]["current_step"] = "Import Neo4j hoàn thành!"
        extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        extraction_jobs[job_id]["stats"] = {
            "source_file": filename,
            "category": category,
            "imported": {
                "documents": stats.documents,
                "chapters": stats.chapters,
                "articles": stats.articles,
                "clauses": stats.clauses,
                "entities": stats.entities,
                "entities_merged": stats.entities_merged,
                "structural_relations": stats.structural_relations,
                "semantic_relations": stats.semantic_relations
            },
            "neo4j_stats": db_stats
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Neo4j import error: {error_trace}")
        
        extraction_jobs[job_id]["status"] = "failed"
        extraction_jobs[job_id]["error"] = str(e)
        extraction_jobs[job_id]["current_step"] = f"Lỗi: {str(e)}"


@router.post("/neo4j/import")
async def import_json_to_neo4j(
    request: Neo4jImportRequest,
    background_tasks: BackgroundTasks,
    clear_existing: bool = False
):
    """
    Import JSON data directly to Neo4j (via request body instead of file upload).
    """
    job_id = str(uuid.uuid4())
    
    extraction_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "current_step": "Đang chuẩn bị import Neo4j...",
        "stage": "neo4j_import",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result_file": None,
        "error": None,
        "stats": None
    }
    
    background_tasks.add_task(
        run_neo4j_import,
        job_id,
        request.data,
        "api_request",
        clear_existing
    )
    
    return ExtractionStatus(**extraction_jobs[job_id])


@router.get("/neo4j/stats")
async def get_neo4j_stats():
    """Get current Neo4j database statistics."""
    try:
        from app.core.indexing.graph_builder import Neo4jGraphBuilder
        
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        with Neo4jGraphBuilder(uri=uri, user=user, password=password) as builder:
            stats = builder.get_graph_stats()
        
        return {"status": "connected", "stats": stats}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.delete("/neo4j/clear")
async def clear_neo4j():
    """Clear all data from Neo4j database."""
    try:
        from app.core.indexing.graph_builder import Neo4jGraphBuilder
        
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        with Neo4jGraphBuilder(uri=uri, user=user, password=password) as builder:
            builder.clear_database()
        
        return {"status": "success", "message": "Đã xóa toàn bộ dữ liệu Neo4j"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# VECTOR DATABASE (Weaviate) INDEXING
# =============================================================================

@router.post("/weaviate/upload")
async def upload_json_to_weaviate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: str = "regulation"
):
    """
    Upload a JSON file and index to Weaviate vector database.
    """
    if not file.filename.lower().endswith('.json'):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file JSON")
    
    try:
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"File JSON không hợp lệ: {str(e)}")
    
    job_id = str(uuid.uuid4())
    
    extraction_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "current_step": "Đang chuẩn bị index Weaviate...",
        "stage": "weaviate_index",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result_file": None,
        "error": None,
        "stats": None
    }
    
    background_tasks.add_task(
        run_weaviate_index,
        job_id,
        data,
        file.filename,
        doc_type
    )
    
    return ExtractionStatus(**extraction_jobs[job_id])


async def run_weaviate_index(
    job_id: str,
    data: Dict[str, Any],
    filename: str,
    doc_type: str
):
    """Index extraction data to Weaviate."""
    try:
        extraction_jobs[job_id]["status"] = "processing"
        extraction_jobs[job_id]["current_step"] = "Đang xử lý dữ liệu JSON..."
        extraction_jobs[job_id]["progress"] = 10
        
        from app.core.indexing.index_semantic_data import load_and_process_json, convert_to_document_chunks
        from adapters.weaviate_vector_adapter import WeaviateVectorAdapter
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from app.config.settings import settings
        import tempfile
        
        # Save data to temp file for processing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            temp_path = f.name
        
        extraction_jobs[job_id]["progress"] = 20
        extraction_jobs[job_id]["current_step"] = "Đang trích xuất chunks..."
        
        # Process JSON to chunks
        chunks_data = load_and_process_json(temp_path, doc_type)
        
        if not chunks_data:
            raise Exception("Không thể trích xuất chunks từ JSON")
        
        extraction_jobs[job_id]["progress"] = 40
        extraction_jobs[job_id]["current_step"] = f"Đang chuyển đổi {len(chunks_data)} chunks..."
        
        # Convert to DocumentChunk
        doc_filename = Path(filename).stem
        document_chunks = convert_to_document_chunks(chunks_data, doc_filename)
        
        extraction_jobs[job_id]["progress"] = 50
        extraction_jobs[job_id]["current_step"] = "Đang kết nối Weaviate..."
        
        # Initialize embedding model and adapter
        embedding_model = HuggingFaceEmbedding(model_name=settings.emb_model)
        weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8090")
        
        vector_adapter = WeaviateVectorAdapter(
            weaviate_url=weaviate_url,
            embedding_model=embedding_model,
            api_key=None
        )
        
        extraction_jobs[job_id]["progress"] = 60
        extraction_jobs[job_id]["current_step"] = f"Đang index {len(document_chunks)} documents..."
        
        # Index in batches
        batch_size = 50
        total = len(document_chunks)
        indexed = 0
        
        for i in range(0, total, batch_size):
            batch = document_chunks[i:i + batch_size]
            success = await vector_adapter.index_documents(batch)
            if success:
                indexed += len(batch)
            progress = 60 + int((i / total) * 30)
            extraction_jobs[job_id]["progress"] = progress
            extraction_jobs[job_id]["current_step"] = f"Đã index {indexed}/{total} documents..."
        
        # Cleanup temp file
        os.unlink(temp_path)
        
        extraction_jobs[job_id]["progress"] = 100
        extraction_jobs[job_id]["status"] = "completed"
        extraction_jobs[job_id]["current_step"] = "Index Weaviate hoàn thành!"
        extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        extraction_jobs[job_id]["stats"] = {
            "source_file": filename,
            "doc_type": doc_type,
            "chunks_processed": len(chunks_data),
            "documents_indexed": indexed,
            "embedding_model": settings.emb_model
        }
        
    except Exception as e:
        import traceback
        print(f"Weaviate index error: {traceback.format_exc()}")
        extraction_jobs[job_id]["status"] = "failed"
        extraction_jobs[job_id]["error"] = str(e)
        extraction_jobs[job_id]["current_step"] = f"Lỗi: {str(e)}"


@router.get("/weaviate/stats")
async def get_weaviate_stats():
    """Get Weaviate database statistics."""
    try:
        from infrastructure.store.vector.weaviate_store import get_weaviate_client, DOCUMENT_COLLECTION
        
        client = get_weaviate_client()
        collection = client.collections.get(DOCUMENT_COLLECTION)
        
        # Get count
        response = collection.aggregate.over_all(total_count=True)
        total_count = response.total_count
        
        return {
            "status": "connected",
            "stats": {
                "collection": DOCUMENT_COLLECTION,
                "total_documents": total_count
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# OPENSEARCH (BM25) INDEXING
# =============================================================================

@router.post("/opensearch/upload")
async def upload_json_to_opensearch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: str = "regulation",
    clear_existing: bool = False
):
    """
    Upload a JSON file and index to OpenSearch (BM25 keyword search).
    """
    if not file.filename.lower().endswith('.json'):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file JSON")
    
    try:
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"File JSON không hợp lệ: {str(e)}")
    
    job_id = str(uuid.uuid4())
    
    extraction_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "current_step": "Đang chuẩn bị index OpenSearch...",
        "stage": "opensearch_index",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result_file": None,
        "error": None,
        "stats": None
    }
    
    background_tasks.add_task(
        run_opensearch_index,
        job_id,
        data,
        file.filename,
        doc_type,
        clear_existing
    )
    
    return ExtractionStatus(**extraction_jobs[job_id])


async def run_opensearch_index(
    job_id: str,
    data: Dict[str, Any],
    filename: str,
    doc_type: str,
    clear_existing: bool
):
    """Index extraction data to OpenSearch."""
    try:
        extraction_jobs[job_id]["status"] = "processing"
        extraction_jobs[job_id]["current_step"] = "Đang xử lý dữ liệu JSON..."
        extraction_jobs[job_id]["progress"] = 10
        
        from app.core.indexing.index_opensearch_data import load_and_process_json, convert_to_opensearch_documents
        from infrastructure.store.opensearch.client import OpenSearchClient
        import tempfile
        
        # Save data to temp file for processing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            temp_path = f.name
        
        extraction_jobs[job_id]["progress"] = 20
        extraction_jobs[job_id]["current_step"] = "Đang trích xuất chunks..."
        
        # Process JSON to chunks
        chunks_data = load_and_process_json(temp_path, doc_type)
        
        if not chunks_data:
            raise Exception("Không thể trích xuất chunks từ JSON")
        
        extraction_jobs[job_id]["progress"] = 40
        extraction_jobs[job_id]["current_step"] = f"Đang chuyển đổi {len(chunks_data)} chunks..."
        
        # Convert to OpenSearch documents
        doc_filename = Path(filename).stem
        documents = convert_to_opensearch_documents(chunks_data, doc_filename)
        
        extraction_jobs[job_id]["progress"] = 50
        extraction_jobs[job_id]["current_step"] = "Đang kết nối OpenSearch..."
        
        # Initialize OpenSearch client
        client = OpenSearchClient()
        
        # Clear existing if requested
        if clear_existing:
            extraction_jobs[job_id]["current_step"] = "Đang xóa dữ liệu cũ..."
            try:
                delete_result = client.client.delete_by_query(
                    index=client.index_name,
                    body={"query": {"term": {"doc_id": doc_filename}}}
                )
                deleted = delete_result.get("deleted", 0)
                print(f"Deleted {deleted} existing documents")
            except Exception as e:
                print(f"Warning: Could not clear existing: {e}")
        
        extraction_jobs[job_id]["progress"] = 60
        extraction_jobs[job_id]["current_step"] = f"Đang index {len(documents)} documents..."
        
        # Index in batches
        batch_size = 100
        total = len(documents)
        total_success = 0
        total_failed = 0
        
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            try:
                success, failed = client.bulk_index_documents(batch)
                total_success += success
                total_failed += failed
            except Exception as e:
                total_failed += len(batch)
                print(f"Batch error: {e}")
            
            progress = 60 + int((i / total) * 30)
            extraction_jobs[job_id]["progress"] = progress
            extraction_jobs[job_id]["current_step"] = f"Đã index {total_success}/{total} documents..."
        
        # Cleanup temp file
        os.unlink(temp_path)
        
        extraction_jobs[job_id]["progress"] = 100
        extraction_jobs[job_id]["status"] = "completed"
        extraction_jobs[job_id]["current_step"] = "Index OpenSearch hoàn thành!"
        extraction_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        extraction_jobs[job_id]["stats"] = {
            "source_file": filename,
            "doc_type": doc_type,
            "chunks_processed": len(chunks_data),
            "documents_indexed": total_success,
            "documents_failed": total_failed,
            "index_name": client.index_name
        }
        
    except Exception as e:
        import traceback
        print(f"OpenSearch index error: {traceback.format_exc()}")
        extraction_jobs[job_id]["status"] = "failed"
        extraction_jobs[job_id]["error"] = str(e)
        extraction_jobs[job_id]["current_step"] = f"Lỗi: {str(e)}"


@router.get("/opensearch/stats")
async def get_opensearch_stats():
    """Get OpenSearch index statistics."""
    try:
        from infrastructure.store.opensearch.client import OpenSearchClient
        
        client = OpenSearchClient()
        
        # Get index stats
        stats = client.client.indices.stats(index=client.index_name)
        doc_count = stats['indices'][client.index_name]['primaries']['docs']['count']
        
        return {
            "status": "connected",
            "stats": {
                "index_name": client.index_name,
                "total_documents": doc_count
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}