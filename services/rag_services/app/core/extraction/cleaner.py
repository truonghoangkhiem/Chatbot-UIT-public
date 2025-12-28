"""
Post-Processing Cleaner for Stage 1 & Stage 2 Extraction Results.

This module provides functions to clean, validate, and fix common extraction errors:
1. Self-reference modifications (hallucination)
2. Duplicate/Ghost nodes
3. Content bleeding across pages
4. Orphan relations

Author: KG Extraction Team
Date: 2024
"""

import json
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CleaningStats:
    """Statistics from the cleaning process."""
    original_nodes: int = 0
    cleaned_nodes: int = 0
    duplicate_nodes_removed: int = 0
    orphan_relations_removed: int = 0
    invalid_modifications_removed: int = 0
    content_bleeding_fixed: int = 0
    is_original_document: bool = False  # True if this is NOT an amendment
    errors: List[str] = field(default_factory=list)


# =============================================================================
# Document Type Detection
# =============================================================================

def is_amendment_document(data: Dict) -> bool:
    """
    Detect if a document is an amendment (văn bản sửa đổi) or an original (văn bản gốc).
    
    Returns:
        True if this is an amendment document, False if original
    """
    # Check document title/content for amendment indicators
    structure = data.get("structure", {})
    doc = structure.get("document", {})
    
    title = doc.get("title", "") or ""
    full_text = doc.get("full_text", "") or ""
    source_file = data.get("source_file", "") or ""
    
    combined_text = f"{title} {full_text} {source_file}".lower()
    
    # Amendment indicators (văn bản sửa đổi)
    amendment_patterns = [
        r'sửa đổi',
        r'bổ sung',
        r'cập nhật',
        r'thay thế',
        r'điều chỉnh',
        r'1393',  # Known amendment document
    ]
    
    for pattern in amendment_patterns:
        if re.search(pattern, combined_text, re.IGNORECASE):
            return True
    
    # Check if document references another document to amend
    # e.g., "Sửa đổi Quyết định 790"
    if re.search(r'(sửa đổi|bổ sung|cập nhật).{0,50}quyết định.{0,10}\d+', combined_text, re.IGNORECASE):
        return True
    
    return False


# =============================================================================
# Modification Validation
# =============================================================================

def extract_document_signature(data: Dict) -> Optional[str]:
    """
    Extract the document signature/ID from the extraction result.
    
    Looks for patterns like:
    - "790/QĐ-ĐHCNTT"
    - "1393/QĐ-ĐHCNTT"
    - Document node with signature in title/content
    """
    # Check structure.document
    structure = data.get("structure", {})
    doc = structure.get("document")
    
    if doc:
        # Check title for signature pattern
        title = doc.get("title", "") or doc.get("full_text", "")
        signature = extract_signature_from_text(title)
        if signature:
            return signature
    
    # Check source_file
    source_file = data.get("source_file", "")
    if source_file:
        # Pattern: 790-qd-dhcntt... -> 790/QĐ-ĐHCNTT
        match = re.search(r'(\d+)[_-]?qd[_-]?dhcntt', source_file, re.IGNORECASE)
        if match:
            return f"{match.group(1)}/QĐ-ĐHCNTT"
    
    return None


def extract_signature_from_text(text: str) -> Optional[str]:
    """Extract document signature pattern from text."""
    if not text:
        return None
    
    # Pattern: 790/QĐ-ĐHCNTT or 1393/QĐ-ĐHCNTT
    patterns = [
        r'(\d+)/QĐ-ĐHCNTT',
        r'(\d+)/qd-dhcntt',
        r'Quyết định (?:số )?(\d+)',
        r'QĐ[- ]?(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num = match.group(1)
            return f"{num}/QĐ-ĐHCNTT"
    
    return None


def is_self_reference_modification(
    modification: Dict,
    source_document_signature: str
) -> bool:
    """
    Check if a modification is a self-reference (hallucination).
    
    Returns True if the modification's target document is the same
    as the source document (which is invalid).
    """
    target_sig = modification.get("target_document_signature", "")
    
    if not target_sig or not source_document_signature:
        return False
    
    # Normalize for comparison
    source_num = re.search(r'(\d+)', source_document_signature)
    target_num = re.search(r'(\d+)', target_sig)
    
    if source_num and target_num:
        return source_num.group(1) == target_num.group(1)
    
    return target_sig.lower() == source_document_signature.lower()


def remove_invalid_modifications(
    data: Dict,
    source_signature: Optional[str] = None,
    force_is_original: Optional[bool] = None
) -> Tuple[Dict, int, bool]:
    """
    Remove invalid modifications (self-references, missing targets, etc.)
    
    If the document is detected as an ORIGINAL (not amendment), 
    ALL modifications are removed since originals don't modify anything.
    
    Args:
        data: The extraction result data
        source_signature: Optional document signature
        force_is_original: If True, treat as original and remove all modifications
    
    Returns:
        Tuple of (cleaned data, number of modifications removed, is_original_document)
    """
    if source_signature is None:
        source_signature = extract_document_signature(data)
    
    # Detect document type
    is_original = force_is_original if force_is_original is not None else not is_amendment_document(data)
    
    removed_count = 0
    
    # If this is an ORIGINAL document, it should NOT have any modifications
    # (originals don't modify anything - they ARE the base)
    if is_original:
        logger.info(f"Document {source_signature} detected as ORIGINAL - removing ALL modifications")
    
    # Clean modifications in structure.articles
    structure = data.get("structure", {})
    articles = structure.get("articles", [])
    
    for article in articles:
        modifications = article.get("modifications", [])
        if not modifications:
            continue
        
        if is_original:
            # Original document: remove ALL modifications
            removed_count += len(modifications)
            article["modifications"] = []
        else:
            # Amendment document: only remove invalid ones
            valid_mods = []
            for mod in modifications:
                if source_signature and is_self_reference_modification(mod, source_signature):
                    logger.warning(
                        f"Removing self-reference modification: {mod.get('target_article')} "
                        f"-> {mod.get('target_document_signature')}"
                    )
                    removed_count += 1
                elif not mod.get("target_document_signature"):
                    logger.warning(f"Removing modification with no target document: {mod}")
                    removed_count += 1
                else:
                    valid_mods.append(mod)
            article["modifications"] = valid_mods
    
    # Also clean top-level modifications if present
    if "modifications" in data:
        modifications = data.get("modifications", [])
        if is_original:
            removed_count += len(modifications)
            data["modifications"] = []
        else:
            valid_mods = []
            for mod in modifications:
                if source_signature and is_self_reference_modification(mod, source_signature):
                    removed_count += 1
                elif not mod.get("target_document_signature"):
                    removed_count += 1
                else:
                    valid_mods.append(mod)
            data["modifications"] = valid_mods
    
    # Clean stage2_semantic.modifications (merged results)
    stage2 = data.get("stage2_semantic", {})
    if "modifications" in stage2:
        modifications = stage2.get("modifications", [])
        if is_original:
            # Original document: remove ALL modifications
            logger.info(f"Removing ALL {len(modifications)} modifications from stage2_semantic (original document)")
            removed_count += len(modifications)
            stage2["modifications"] = []
        else:
            # Amendment document: only remove invalid ones
            valid_mods = []
            for mod in modifications:
                if source_signature and is_self_reference_modification(mod, source_signature):
                    logger.warning(
                        f"Removing self-reference modification in stage2_semantic: "
                        f"{mod.get('target_article')} -> {mod.get('target_document_signature')}"
                    )
                    removed_count += 1
                elif not mod.get("target_document_signature"):
                    logger.warning(f"Removing modification with no target in stage2_semantic: {mod}")
                    removed_count += 1
                else:
                    valid_mods.append(mod)
            stage2["modifications"] = valid_mods
        data["stage2_semantic"] = stage2
    
    return data, removed_count, is_original


# =============================================================================
# Node Deduplication
# =============================================================================

def normalize_node_id(node_id: str) -> str:
    """
    Normalize node ID to canonical form.
    
    Examples:
    - "chuong_2_to_chuc_dao_tao" -> "chuong_2"
    - "dieu_7_chuong_trinh_dao_tao" -> "dieu_7"
    - "CHUONG_II" -> "chuong_2"
    """
    node_id = node_id.lower().strip()
    
    # Roman numeral to Arabic
    roman_map = {
        'i': '1', 'ii': '2', 'iii': '3', 'iv': '4', 'v': '5',
        'vi': '6', 'vii': '7', 'viii': '8', 'ix': '9', 'x': '10',
    }
    
    # Pattern: chuong_X_extra_text -> chuong_X
    patterns = [
        (r'^(chuong)_([ivxlcdm]+)_.*$', r'\1_\2'),
        (r'^(chuong)_(\d+)_.*$', r'\1_\2'),
        (r'^(dieu)_(\d+)_.*$', r'\1_\2'),
        (r'^(khoan)_(\d+)_(dieu)_(\d+)_.*$', r'\1_\2_\3_\4'),
    ]
    
    for pattern, replacement in patterns:
        if re.match(pattern, node_id):
            node_id = re.sub(pattern, replacement, node_id)
            break
    
    # Convert Roman numerals
    for roman, arabic in roman_map.items():
        node_id = re.sub(f'^(chuong)_{roman}$', f'\\1_{arabic}', node_id)
    
    return node_id


def deduplicate_nodes(nodes: List[Dict]) -> Tuple[List[Dict], int, Dict[str, str]]:
    """
    Remove duplicate nodes based on normalized IDs.
    
    Returns:
        Tuple of (deduplicated nodes, count removed, id_mapping)
    """
    seen: Dict[str, Dict] = {}
    id_mapping: Dict[str, str] = {}  # old_id -> normalized_id
    
    for node in nodes:
        original_id = node.get("id", "")
        norm_id = normalize_node_id(original_id)
        id_mapping[original_id] = norm_id
        
        if norm_id not in seen:
            node["id"] = norm_id
            seen[norm_id] = node
        else:
            # Keep the one with more content
            existing = seen[norm_id]
            existing_len = len(existing.get("full_text", "") or "")
            new_len = len(node.get("full_text", "") or "")
            
            if new_len > existing_len:
                node["id"] = norm_id
                seen[norm_id] = node
                logger.info(f"Replaced duplicate node '{original_id}' with longer content")
    
    removed_count = len(nodes) - len(seen)
    return list(seen.values()), removed_count, id_mapping


def fix_relations_after_dedup(
    relations: List[Dict],
    id_mapping: Dict[str, str],
    valid_node_ids: set
) -> Tuple[List[Dict], int]:
    """
    Fix relations to use normalized IDs and remove orphans.
    
    Returns:
        Tuple of (fixed relations, orphans removed count)
    """
    seen = set()
    fixed = []
    orphans_removed = 0
    
    for rel in relations:
        source = rel.get("source", "")
        target = rel.get("target", "")
        rel_type = rel.get("type", "")
        
        # Normalize IDs
        norm_source = id_mapping.get(source, normalize_node_id(source))
        norm_target = id_mapping.get(target, normalize_node_id(target))
        
        # Check if valid (both nodes exist)
        if norm_source not in valid_node_ids or norm_target not in valid_node_ids:
            orphans_removed += 1
            continue
        
        # Deduplicate
        key = (norm_source, norm_target, rel_type)
        if key not in seen:
            seen.add(key)
            fixed.append({
                "source": norm_source,
                "target": norm_target,
                "type": rel_type
            })
    
    return fixed, orphans_removed


# =============================================================================
# Content Bleeding Detection & Fix
# =============================================================================

def detect_content_bleeding(articles: List[Dict]) -> List[Dict]:
    """
    Detect articles where content appears to belong to a different article.
    
    Heuristics:
    - Article about "Học phí" has content about "Chương trình đào tạo"
    - Full_text starts with content unrelated to title
    """
    issues = []
    
    keywords_map = {
        "học phí": ["học phí", "mức phí", "thanh toán", "hphk", "đóng phí"],
        "chương trình đào tạo": ["chương trình", "học phần", "tín chỉ", "đào tạo"],
        "tuyển sinh": ["tuyển sinh", "xét tuyển", "đăng ký", "hồ sơ"],
        "tốt nghiệp": ["tốt nghiệp", "bằng", "xét tốt nghiệp", "bảo vệ"],
    }
    
    for article in articles:
        title = (article.get("title", "") or "").lower()
        content = (article.get("full_text", "") or "").lower()[:500]
        
        # Find expected topic from title
        expected_topic = None
        for topic, keywords in keywords_map.items():
            if any(kw in title for kw in keywords):
                expected_topic = topic
                break
        
        if not expected_topic:
            continue
        
        # Check if content matches a DIFFERENT topic
        for other_topic, other_keywords in keywords_map.items():
            if other_topic == expected_topic:
                continue
            
            # If content has many keywords from a different topic
            matches = sum(1 for kw in other_keywords if kw in content)
            if matches >= 2:
                issues.append({
                    "article_id": article.get("id"),
                    "expected_topic": expected_topic,
                    "detected_topic": other_topic,
                    "severity": "high" if matches >= 3 else "medium"
                })
                break
    
    return issues


def is_truncated_content(text: str) -> bool:
    """Check if text appears to be truncated mid-sentence."""
    if not text:
        return False
    
    text = text.strip()
    
    # Ends with incomplete word/sentence
    truncation_patterns = [
        r'\s+\w{1,3}$',  # Ends with 1-3 char word (likely incomplete)
        r'[,;:]\s*$',    # Ends with comma/semicolon
        r'\.\.\.$',      # Ends with ...
        r'bằng với\s*$', # Common truncation pattern
        r'\s+là\s*$',
        r'\s+và\s*$',
        r'\s+của\s*$',
    ]
    
    for pattern in truncation_patterns:
        if re.search(pattern, text):
            return True
    
    return False


# =============================================================================
# Main Cleaning Function
# =============================================================================

def clean_extraction_result(
    data: Dict,
    source_signature: Optional[str] = None,
    remove_self_ref_mods: bool = True,
    deduplicate: bool = True,
    fix_relations: bool = True
) -> Tuple[Dict, CleaningStats]:
    """
    Clean and validate extraction result.
    
    Performs:
    1. Remove self-reference modifications
    2. Deduplicate nodes
    3. Fix and validate relations
    4. Detect content bleeding issues
    
    Args:
        data: Extraction result dictionary
        source_signature: Document signature (auto-detected if None)
        remove_self_ref_mods: Whether to remove self-reference modifications
        deduplicate: Whether to deduplicate nodes
        fix_relations: Whether to fix relations
        
    Returns:
        Tuple of (cleaned data, cleaning statistics)
    """
    stats = CleaningStats()
    
    # Auto-detect source signature
    if source_signature is None:
        source_signature = extract_document_signature(data)
        if source_signature:
            logger.info(f"Detected source document: {source_signature}")
    
    structure = data.get("structure", {})
    
    # Count original nodes
    for node_type in ["chapters", "articles", "clauses", "tables"]:
        nodes = structure.get(node_type, [])
        stats.original_nodes += len(nodes)
    
    # Step 1: Remove invalid modifications
    if remove_self_ref_mods:
        data, removed, is_original = remove_invalid_modifications(data, source_signature)
        stats.invalid_modifications_removed = removed
        stats.is_original_document = is_original
        if removed > 0:
            logger.info(f"Removed {removed} invalid modifications (is_original={is_original})")
    
    # Step 2: Deduplicate nodes
    if deduplicate:
        all_id_mapping = {}
        all_valid_ids = set()
        
        for node_type in ["chapters", "articles", "clauses", "tables"]:
            nodes = structure.get(node_type, [])
            if nodes:
                deduped, removed, id_map = deduplicate_nodes(nodes)
                structure[node_type] = deduped
                stats.duplicate_nodes_removed += removed
                all_id_mapping.update(id_map)
                all_valid_ids.update(n["id"] for n in deduped)
        
        if stats.duplicate_nodes_removed > 0:
            logger.info(f"Removed {stats.duplicate_nodes_removed} duplicate nodes")
        
        # Step 3: Fix relations
        if fix_relations:
            relations = structure.get("relations", [])
            if relations:
                fixed, orphans = fix_relations_after_dedup(
                    relations, all_id_mapping, all_valid_ids
                )
                structure["relations"] = fixed
                stats.orphan_relations_removed = orphans
                if orphans > 0:
                    logger.info(f"Removed {orphans} orphan relations")
    
    # Step 4: Detect content bleeding
    articles = structure.get("articles", [])
    bleeding_issues = detect_content_bleeding(articles)
    if bleeding_issues:
        stats.errors.extend([
            f"Content bleeding in {issue['article_id']}: "
            f"expected '{issue['expected_topic']}', found '{issue['detected_topic']}'"
            for issue in bleeding_issues
        ])
        logger.warning(f"Detected {len(bleeding_issues)} content bleeding issues")
    
    # Step 5: Detect truncated content
    for article in articles:
        if is_truncated_content(article.get("full_text", "")):
            stats.errors.append(
                f"Truncated content in {article.get('id')}: "
                f"'{article.get('full_text', '')[-30:]}...'"
            )
    
    # Count final nodes
    for node_type in ["chapters", "articles", "clauses", "tables"]:
        nodes = structure.get(node_type, [])
        stats.cleaned_nodes += len(nodes)
    
    data["structure"] = structure
    data["cleaning_stats"] = {
        "original_nodes": stats.original_nodes,
        "cleaned_nodes": stats.cleaned_nodes,
        "duplicates_removed": stats.duplicate_nodes_removed,
        "orphan_relations_removed": stats.orphan_relations_removed,
        "invalid_modifications_removed": stats.invalid_modifications_removed,
        "issues_detected": len(stats.errors)
    }
    
    return data, stats


def clean_json_file(
    input_path: str,
    output_path: Optional[str] = None,
    **kwargs
) -> CleaningStats:
    """
    Clean a JSON extraction result file.
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output file (default: input_path with _cleaned suffix)
        **kwargs: Additional arguments for clean_extraction_result
        
    Returns:
        Cleaning statistics
    """
    input_path = Path(input_path)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
    
    logger.info(f"Cleaning extraction result: {input_path}")
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    cleaned_data, stats = clean_extraction_result(data, **kwargs)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Cleaned result saved to: {output_path}")
    logger.info(f"Stats: {stats.original_nodes} -> {stats.cleaned_nodes} nodes")
    
    if stats.errors:
        logger.warning(f"Issues detected ({len(stats.errors)}):")
        for error in stats.errors[:5]:  # Show first 5
            logger.warning(f"  - {error}")
    
    return stats


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    parser = argparse.ArgumentParser(
        description="Clean extraction result JSON files"
    )
    parser.add_argument("input", help="Input JSON file path")
    parser.add_argument("-o", "--output", help="Output JSON file path")
    parser.add_argument(
        "--source-sig",
        help="Source document signature (e.g., '790/QĐ-ĐHCNTT')"
    )
    
    args = parser.parse_args()
    
    stats = clean_json_file(
        args.input,
        args.output,
        source_signature=args.source_sig
    )
    
    print(f"\n{'='*50}")
    print("CLEANING COMPLETE")
    print(f"{'='*50}")
    print(f"Original nodes: {stats.original_nodes}")
    print(f"Cleaned nodes: {stats.cleaned_nodes}")
    print(f"Duplicates removed: {stats.duplicate_nodes_removed}")
    print(f"Orphan relations removed: {stats.orphan_relations_removed}")
    print(f"Invalid modifications removed: {stats.invalid_modifications_removed}")
    print(f"Issues detected: {len(stats.errors)}")
