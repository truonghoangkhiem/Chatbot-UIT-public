"""
Hybrid Two-Stage Knowledge Graph Extraction Pipeline for Legal Documents.

This module implements a two-stage extraction pipeline:
    - Stage 1 (Structural Extraction): Uses VLM to extract document structure and tables.
    - Stage 2 (Semantic Extraction): Uses LLM to extract semantic entities.

Features:
- Robust Table Merging: Appends Markdown tables to parent Articles before semantic extraction.
- Parallel Processing: Uses asyncio for fast extraction.
- DEBUG MODE: Includes print statements for troubleshooting.
"""

import asyncio
import base64
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# Add rag_services root to path for imports
RAG_SERVICES_ROOT = Path(__file__).parent.parent.parent.parent
if str(RAG_SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_SERVICES_ROOT))

from app.core.utils.json_utils import clean_and_parse_json
from app.core.extraction.page_merger import merge_nodes_into_dict

# Import updated schemas
from app.core.extraction.schemas import (
    StructureNodeType, VLMProvider, StructureNode, StructureRelation, 
    StructureExtractionResult, SemanticNode, SemanticRelation, Modification,
    SemanticExtractionResult, HybridExtractionResult, PageContext, 
    VLMConfig, LLMConfig, VALID_ENTITY_TYPES, VALID_RELATION_TYPES, 
    UNIFIED_ACADEMIC_SCHEMA, STRUCTURE_EXTRACTION_PROMPT, 
    SEMANTIC_EXTRACTION_PROMPT
)

from core.domain.graph_models import (
    GraphNode, GraphRelationship, NodeCategory, RelationshipType
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Stage 1: Structure Extractor (VLM)
# =============================================================================

class StructureExtractor:
    def __init__(self, config: VLMConfig):
        self.config = config
        logger.info(f"StructureExtractor initialized with {config.provider.value}/{config.model}")
    
    def _encode_image(self, image_path: Path) -> Tuple[str, str]:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        extension = image_path.suffix.lower()
        media_type = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(extension, "image/png")
        return image_base64, media_type
    
    def _call_vlm_api(self, image_base64: str, media_type: str, prev_context: PageContext, page_number: int) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required.")
        
        prompt = STRUCTURE_EXTRACTION_PROMPT.format(prev_context=prev_context.model_dump_json(indent=2))
        
        user_content = [
            {"type": "text", "text": f"Page {page_number} context.\n\n{prompt}"},
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}", "detail": "high"}}
        ]
        
        client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        response = client.chat.completions.create(
            model=self.config.model, messages=[{"role": "user", "content": user_content}],
            max_tokens=self.config.max_tokens, temperature=self.config.temperature
        )
        return response.choices[0].message.content
    
    def _parse_vlm_response(self, response_text: str, page_number: int) -> Dict[str, Any]:
        if not response_text: return {}
        data, errors = clean_and_parse_json(response_text, logger)
        if errors: logger.warning(f"Page {page_number} JSON errors: {errors}")
        return data if isinstance(data, dict) else {}
    
    def _process_single_page(self, image_path: Path, prev_context: PageContext, page_number: int):
        image_base64, media_type = self._encode_image(image_path)
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response_text = self._call_vlm_api(image_base64, media_type, prev_context, page_number)
                data = self._parse_vlm_response(response_text, page_number)
                
                nodes = []
                for n in data.get("nodes", []):
                    try:
                        # Normalize type casing and map to Enum
                        t_str = n.get("type", "").capitalize()
                        try:
                            n_type = StructureNodeType(t_str)
                        except ValueError:
                            # Fallback logic
                            if "table" in t_str.lower():
                                n_type = StructureNodeType.TABLE
                            else:
                                n_type = StructureNodeType.CLAUSE
                                
                        nodes.append(StructureNode(
                            id=n["id"], type=n_type,
                            title=n.get("title", ""), full_text=n.get("full_text", ""),
                            page_range=[page_number], metadata={"source_page": page_number}
                        ))
                    except Exception as e:
                        logger.warning(f"Skipping malformed node: {e}")

                relations = [StructureRelation(**r) for r in data.get("relations", [])]
                next_context = PageContext(**data.get("next_context", {}))
                return nodes, relations, next_context
            except Exception as e:
                logger.warning(f"Page {page_number} attempt {attempt} failed: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay)
        return [], [], prev_context

    def extract_from_images(self, image_paths: List[Path], continue_on_error: bool = True) -> StructureExtractionResult:
        merged_nodes = {}
        all_relations = []
        errors = []
        current_context = PageContext()
        
        last_table_of_prev_page = None

        for i, path in enumerate(image_paths, 1):
            try:
                nodes, relations, next_context = self._process_single_page(Path(path), current_context, i)
                
                # --- LOGIC XỬ LÝ BẢNG BỊ NGẮT (BACKWARD MERGE - MẠNH MẼ HƠN) ---
                if nodes and last_table_of_prev_page:
                    # Thay vì chỉ kiểm tra node[0], ta kiểm tra 3 node đầu tiên
                    # để tránh trường hợp có header/số trang rác chèn vào
                    found_fragment_idx = -1
                    
                    for idx, node in enumerate(nodes[:3]):
                        is_fragment = False
                        # Dấu hiệu 1: Là Table và (không có title hoặc title trùng/chứa từ "tiếp")
                        if node.type == StructureNodeType.TABLE:
                             if not node.title or node.title == last_table_of_prev_page.title:
                                is_fragment = True
                             elif "tiếp" in (node.title or "").lower() or "cont" in (node.title or "").lower():
                                is_fragment = True
                        
                        # Dấu hiệu 2: Text bắt đầu bằng "|" (Markdown Table)
                        elif node.full_text.strip().startswith("|"):
                             is_fragment = True
                        
                        if is_fragment:
                            found_fragment_idx = idx
                            break
                    
                    if found_fragment_idx != -1:
                        fragment_node = nodes[found_fragment_idx]
                        logger.info(f"MERGING SPLIT TABLE: Page {i} (Node {found_fragment_idx}) -> Table '{last_table_of_prev_page.id}'")
                        
                        new_content = fragment_node.full_text.strip()
                        lines = new_content.split('\n')
                        prev_lines = last_table_of_prev_page.full_text.strip().split('\n')
                        
                        # Bỏ header bị lặp (nếu có)
                        if len(lines) > 0 and len(prev_lines) > 0:
                             # Check dòng 1
                             if lines[0] in prev_lines[:10]: # Quét sâu hơn trong bảng cũ
                                new_content = "\n".join(lines[1:])
                                # Check dòng 2 (nếu là dòng kẻ phân cách |---|)
                                if new_content.strip().startswith("|-"):
                                     new_content = "\n".join(new_content.split('\n')[1:])

                        # NỐI VÀO BẢNG CŨ
                        last_table_of_prev_page.full_text += "\n" + new_content
                        if i not in last_table_of_prev_page.page_range:
                            last_table_of_prev_page.page_range.append(i)
                        
                        # Xóa node mảnh thừa khỏi danh sách
                        nodes.pop(found_fragment_idx)

                # Cập nhật bảng cuối cùng để dùng cho trang sau
                tables_in_page = [n for n in nodes if n.type == StructureNodeType.TABLE]
                if tables_in_page:
                    last_table_of_prev_page = tables_in_page[-1]
                elif nodes: 
                    # Nếu trang này có nội dung (Article, Clause) chen vào thì ngắt mạch nối
                    # Trừ khi node đó bị nhận diện nhầm là fragment ở trên và đã bị pop()
                    last_table_of_prev_page = None
                
                merged_nodes = merge_nodes_into_dict(merged_nodes, nodes, i, logger)
                all_relations.extend(relations)
                current_context = next_context
            except Exception as e:
                errors.append(f"Page {i}: {e}")
                if not continue_on_error: raise

        all_nodes = list(merged_nodes.values())
        return StructureExtractionResult(
            document=next((n for n in all_nodes if n.type == StructureNodeType.DOCUMENT), None),
            chapters=[n for n in all_nodes if n.type == StructureNodeType.CHAPTER],
            articles=[n for n in all_nodes if n.type == StructureNodeType.ARTICLE],
            clauses=[n for n in all_nodes if n.type in [StructureNodeType.CLAUSE, StructureNodeType.POINT]],
            tables=[n for n in all_nodes if n.type == StructureNodeType.TABLE],
            relations=all_relations,
            page_count=len(image_paths),
            errors=errors
        )

    def extract_from_pdf(self, pdf_path: Path, output_dir: Optional[Path] = None, keep_images: bool = True) -> StructureExtractionResult:
        from pdf2image import convert_from_path
        output_dir = output_dir or pdf_path.parent / f"{pdf_path.stem}_images"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        images = convert_from_path(pdf_path, dpi=200, fmt="png")
        image_paths = []
        for i, img in enumerate(images, 1):
            p = Path(output_dir) / f"page_{i:03d}.png"
            img.save(p, "PNG")
            image_paths.append(p)
            
        result = self.extract_from_images(image_paths)
        if not keep_images:
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
        return result


# =============================================================================
# Stage 2: Semantic Extractor (Classes)
# =============================================================================

class SemanticExtractor:
    """
    Standard Semantic Extractor (Synchronous/Single processing).
    Used as a base for ParallelSemanticExtractor and for backward compatibility.
    """
    def __init__(self, config: LLMConfig):
        self.config = config
        logger.info(f"SemanticExtractor initialized with {config.model}")
    
    def _call_llm_api(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai required")
        
        client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        response = client.chat.completions.create(
            model=self.config.model, messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens, temperature=self.config.temperature
        )
        return response.choices[0].message.content
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        data, errors = clean_and_parse_json(response_text, logger)
        if not isinstance(data, dict): return {"entities": [], "relations": [], "modifications": []}
        return data
    
    def _post_process_extraction(self, data: Dict[str, Any], parent_article_id: str) -> Dict[str, Any]:
        id_mapping = {}
        processed_entities = []
        # Normalize Entity IDs
        for ent in data.get("entities", []):
            old_id = str(ent.get("id", "unknown"))
            new_id = f"{parent_article_id}_ent_{old_id}"
            id_mapping[old_id] = new_id
            processed_entities.append({**ent, "id": new_id})
            
        # Update Relation IDs
        processed_relations = []
        for rel in data.get("relations", []):
            src, tgt = str(rel.get("source_id")), str(rel.get("target_id"))
            new_src = id_mapping.get(src, f"{parent_article_id}_ent_{src}")
            new_tgt = id_mapping.get(tgt, f"{parent_article_id}_ent_{tgt}")
            processed_relations.append({**rel, "source_id": new_src, "target_id": new_tgt})
        
        # Process modifications - ensure source_text_id is set
        processed_modifications = []
        for mod in data.get("modifications", []):
            mod_copy = {**mod}
            if not mod_copy.get("source_text_id"):
                mod_copy["source_text_id"] = parent_article_id
            processed_modifications.append(mod_copy)
            
        return {"entities": processed_entities, "relations": processed_relations, "modifications": processed_modifications}

    def _validate_entity_type(self, entity_type: str) -> bool:
        return entity_type.upper() in VALID_ENTITY_TYPES

    def _validate_relation_type(self, rel_type: str) -> bool:
        return rel_type.upper() in VALID_RELATION_TYPES
    
    def _detect_amendment_pattern(self, article_title: str, article_text: str) -> Optional[str]:
        """
        Detect if this article contains amendment patterns like "Khoản X Điều Y".
        Returns a hint string to append to prompt if detected.
        """
        import re
        
        # Patterns indicating amendments
        amendment_keywords = ["sửa đổi", "bổ sung", "cập nhật", "thay thế", "điều chỉnh"]
        clause_article_pattern = r"[Kk]hoản\s+\d+\s+[Đđ]iều\s+\d+"
        point_pattern = r"[Đđ]iểm\s+[a-z]\s+[Kk]hoản\s+\d+\s+[Đđ]iều\s+\d+"
        article_ref_pattern = r"[Qq]uyết\s+định\s+(?:số\s+)?(\d+)"
        
        combined_text = f"{article_title} {article_text}".lower()
        
        # Check for amendment keywords
        has_amendment_keyword = any(kw in combined_text for kw in amendment_keywords)
        
        # Check for "Khoản X Điều Y" patterns
        clause_matches = re.findall(clause_article_pattern, article_title + " " + article_text, re.IGNORECASE)
        point_matches = re.findall(point_pattern, article_title + " " + article_text, re.IGNORECASE)
        
        # Check for target document reference
        doc_refs = re.findall(article_ref_pattern, article_title + " " + article_text, re.IGNORECASE)
        target_doc = doc_refs[0] if doc_refs else "790"  # Default to 790 if not found
        
        if clause_matches or point_matches or has_amendment_keyword:
            hint = f"""

**GỢI Ý TỪ HỆ THỐNG**: Đây là văn bản SỬA ĐỔI. Phát hiện các patterns sau:
- Số văn bản gốc được tham chiếu: QĐ {target_doc}/QĐ-ĐHCNTT
- Các điều khoản được sửa đổi: {', '.join(clause_matches[:5]) if clause_matches else 'Không rõ'}
- Các điểm được sửa đổi: {', '.join(point_matches[:3]) if point_matches else 'Không có'}

→ BẮT BUỘC tạo modifications cho mỗi "Khoản X Điều Y" hoặc "Điểm x Khoản Y Điều Z"!
"""
            return hint
        return None

    def extract_from_article(self, article_id: str, article_title: str, article_text: str) -> SemanticExtractionResult:
        # Detect amendment patterns and add hint
        amendment_hint = self._detect_amendment_pattern(article_title, article_text)
        
        prompt = SEMANTIC_EXTRACTION_PROMPT.format(
            schema_definition=UNIFIED_ACADEMIC_SCHEMA,
            article_id=article_id, article_title=article_title, article_text=article_text
        )
        
        # Append amendment hint if detected
        if amendment_hint:
            prompt += amendment_hint
            logger.info(f"[Stage2] Amendment patterns detected in {article_id}, adding extraction hint")
        
        try:
            # DEBUG: Log input text length
            logger.info(f"[Stage2] Processing {article_id}: {len(article_text)} chars, title: {article_title}")
            
            raw_text = self._call_llm_api(prompt)
            
            # DEBUG: Log raw LLM response
            logger.info(f"[Stage2] LLM response for {article_id}: {raw_text[:500]}...")
            
            data = self._parse_llm_response(raw_text)
            data = self._post_process_extraction(data, article_id)
            
            # DEBUG: Log extraction results
            logger.info(f"[Stage2] Extracted from {article_id}: {len(data.get('entities', []))} entities, {len(data.get('relations', []))} relations, {len(data.get('modifications', []))} modifications")
            
            nodes = [SemanticNode(**e, source_article_id=article_id) for e in data["entities"] if self._validate_entity_type(e.get("type", ""))]
            relations = [SemanticRelation(**r, source_article_id=article_id) for r in data["relations"] if self._validate_relation_type(r.get("type", ""))]
            modifications = [Modification(**m) for m in data.get("modifications", [])]
            
            return SemanticExtractionResult(article_id=article_id, nodes=nodes, relations=relations, modifications=modifications)
        except Exception as e:
            logger.error(f"Error extracting {article_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return SemanticExtractionResult(article_id=article_id, errors=[str(e)])


class ParallelSemanticExtractor:
    """
    Optimized Semantic Extractor using AsyncIO.
    """
    def __init__(self, config: LLMConfig, max_concurrency: int = 5):
        self.config = config
        self.max_concurrency = max_concurrency
        self.semaphore = None
        # Reuse logic from Sync Extractor
        self.sync_extractor = SemanticExtractor(config)

    async def _call_llm(self, prompt: str) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        response = await client.chat.completions.create(
            model=self.config.model, messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_tokens, temperature=self.config.temperature
        )
        return response.choices[0].message.content

    async def _extract_single(self, article: Dict[str, str], pbar=None):
        # Detect amendment patterns and add hint
        amendment_hint = self.sync_extractor._detect_amendment_pattern(article["title"], article["full_text"])
        
        prompt = SEMANTIC_EXTRACTION_PROMPT.format(
            schema_definition=UNIFIED_ACADEMIC_SCHEMA,
            article_id=article["id"], article_title=article["title"], article_text=article["full_text"]
        )
        
        # Append amendment hint if detected
        if amendment_hint:
            prompt += amendment_hint
            logger.info(f"[Stage2-Async] Amendment patterns detected in {article['id']}, adding extraction hint")
        
        async with self.semaphore:
            try:
                raw = await self._call_llm(prompt)
                
                # Reuse parsing/validation logic from sync extractor
                data = self.sync_extractor._parse_llm_response(raw)
                data = self.sync_extractor._post_process_extraction(data, article["id"])
                
                nodes = []
                for e in data.get("entities", []):
                    if self.sync_extractor._validate_entity_type(e.get("type", "")):
                        nodes.append(SemanticNode(**e, source_article_id=article["id"]))
                
                relations = []
                for r in data.get("relations", []):
                    if self.sync_extractor._validate_relation_type(r.get("type", "")):
                        relations.append(SemanticRelation(**r, source_article_id=article["id"]))
                
                modifications = [Modification(**m) for m in data.get("modifications", [])]
                
                if pbar: pbar.update(1)
                return SemanticExtractionResult(article_id=article["id"], nodes=nodes, relations=relations, modifications=modifications)
            except Exception as e:
                return SemanticExtractionResult(article_id=article["id"], errors=[str(e)])

    async def extract_batch_async(self, articles: List[Dict[str, str]], show_progress: bool = True):
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._extract_single(a) for a in articles]
        
        if show_progress:
            try:
                from tqdm.asyncio import tqdm
                return [await t for t in tqdm.as_completed(tasks, total=len(tasks), desc="Semantic Extraction")]
            except ImportError: pass
        return await asyncio.gather(*tasks)

    def extract_batch(self, articles: List[Dict[str, str]], show_progress: bool = True):
        return asyncio.run(self.extract_batch_async(articles, show_progress))


# =============================================================================
# Pipeline Controller & Graph Converters
# =============================================================================

def convert_to_graph_models(hybrid_result: HybridExtractionResult) -> Tuple[List[GraphNode], List[GraphRelationship]]:
    graph_nodes = []
    
    # 1. Articles as Nodes
    for a in hybrid_result.structure.articles:
        graph_nodes.append(GraphNode(
            id=a.id, category=NodeCategory.QUY_DINH,
            properties={"title": a.title, "full_text": a.full_text, "source": "legal_doc"}
        ))
        
    # 2. Semantic Nodes
    for s in hybrid_result.semantic_nodes:
        cat = getattr(NodeCategory, s.type, NodeCategory.DIEU_KIEN)
        graph_nodes.append(GraphNode(
            id=s.id, category=cat,
            properties={"text": s.text, "normalized": s.normalized, "source_article": s.source_article_id}
        ))
        
    # 3. Relations
    graph_rels = []
    for r in hybrid_result.semantic_relations:
        rel_type = getattr(RelationshipType, r.type, RelationshipType.LIEN_QUAN_NOI_DUNG)
        graph_rels.append(GraphRelationship(
            source_id=r.source_id, target_id=r.target_id, rel_type=rel_type,
            properties={"evidence": r.evidence}
        ))
    return graph_nodes, graph_rels

def run_pipeline(image_paths=None, pdf_path=None, vlm_config=None, llm_config=None, output_path=None):
    vlm_config = vlm_config or VLMConfig.from_env()
    llm_config = llm_config or LLMConfig.from_env()
    
    logger.info("--- Starting Pipeline ---")
    
    # 1. STAGE 1: Structure Extraction
    structure_extractor = StructureExtractor(vlm_config)
    if pdf_path:
        structure_result = structure_extractor.extract_from_pdf(Path(pdf_path))
    else:
        structure_result = structure_extractor.extract_from_images([Path(p) for p in image_paths])
    
    # =========================================================================
    # CRITICAL: TABLE MERGING LOGIC WITH DEBUG PRINTS & NORMALIZATION
    # =========================================================================
    
    print("\n--- DEBUG: STARTING TABLE MERGE ---")
    # B1: Map Article ID
    article_map = {a.id.strip(): a for a in structure_result.articles}
    print(f"DEBUG: Loaded {len(article_map)} articles into map.")
    
    # B2: Map Relationship (Child -> Parent)
    child_to_parent = {}
    for rel in structure_result.relations:
        if rel.type == "CONTAINS":
            child_to_parent[rel.target.strip()] = rel.source.strip()
    print(f"DEBUG: Mapped {len(child_to_parent)} child-parent relationships.")
            
    # B3: Merge Tables
    merged_count = 0
    print(f"DEBUG: Found {len(structure_result.tables)} tables to process.")
    
    for table in structure_result.tables:
        table_id = table.id.strip()
        parent_id = child_to_parent.get(table_id)
        
        # Fallback Strategy
        if not parent_id:
            for art_id in article_map.keys():
                if art_id in table_id: 
                    parent_id = art_id
                    print(f"DEBUG: Inferred parent {parent_id} for orphan table {table_id}")
                    break

        if parent_id and parent_id in article_map:
            parent_article = article_map[parent_id]
            
            print(f"DEBUG: MERGING Table '{table_id}' into Article '{parent_id}'")
            logger.info(f"MERGING Table {table_id} into Article {parent_id}")
            
            # Appending Table Markdown to Article Text
            append_text = f"\n\n=== BẢNG THAM CHIẾU ({table.title}) ===\n{table.full_text}\n========================\n"
            parent_article.full_text += append_text
            merged_count += 1
        else:
            print(f"DEBUG: FAILED TO MERGE Table '{table_id}'. Parent ID found: '{parent_id}'")
            logger.warning(f"Orphan Table detected: {table_id}")

    print(f"DEBUG: Successfully merged {merged_count} tables.")
    print("--- DEBUG: END TABLE MERGE ---\n")

    # 2. STAGE 2: Semantic Extraction (Using Enriched Articles)
    semantic_extractor = ParallelSemanticExtractor(llm_config)
    
    # Prepare data with MERGED text
    articles_data = []
    for a in structure_result.articles:
        if a.full_text.strip():
            articles_data.append({"id": a.id, "title": a.title, "full_text": a.full_text})
    
    semantic_results = semantic_extractor.extract_batch(articles_data)
    
    # 3. Result Aggregation
    hybrid_result = HybridExtractionResult(
        structure=structure_result,
        semantic_nodes=[n for r in semantic_results for n in r.nodes],
        semantic_relations=[rel for r in semantic_results for rel in r.relations],
        modifications=[m for r in semantic_results for m in r.modifications],
        total_pages=structure_result.page_count,
        total_articles_processed=len(articles_data),
        errors=structure_result.errors
    )
    
    graph_nodes, graph_rels = convert_to_graph_models(hybrid_result)
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(hybrid_result.model_dump(), f, ensure_ascii=False, indent=2)
            
    logger.info(f"Pipeline Finished. Extracted {len(graph_nodes)} nodes, {len(graph_rels)} relations.")
    return hybrid_result, graph_nodes, graph_rels

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str)
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output", type=str, default="result.json")
    args = parser.parse_args()
    
    try:
        run_pipeline(
            pdf_path=args.pdf, 
            image_paths=sorted(Path(args.input_dir).glob("*.png")) if args.input_dir else None, 
            output_path=args.output
        )
    except Exception as e:
        logger.error(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()