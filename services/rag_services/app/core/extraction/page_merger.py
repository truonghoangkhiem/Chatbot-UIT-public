"""
Cross-Page Node Merging Utilities for Structural Extraction.

Handles the critical problem of content split across multiple pages:
- Merging text when an Article spans multiple pages
- Deduplication while preserving complete content
- Page range tracking

Author: Legal Document Processing Team
Date: 2024
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from copy import deepcopy

logger = logging.getLogger(__name__)


@dataclass
class MergeableNode:
    """
    Node that can be merged across pages.
    
    Designed to handle the case where VLM returns partial content
    from each page that needs to be combined.
    """
    id: str
    type: str
    title: str
    full_text: str
    page_range: List[int] = field(default_factory=list)
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def merge_text(self, additional_text: str, page_number: int) -> None:
        """
        Append additional text from another page.
        
        Args:
            additional_text: Text to append
            page_number: Page where this text came from
        """
        if not additional_text:
            return
        
        # Avoid duplicate content
        if additional_text.strip() in self.full_text:
            logger.debug(f"Skipping duplicate text for {self.id}")
            return
        
        # Smart merge: handle sentence continuation
        old_text = self.full_text.rstrip()
        new_text = additional_text.lstrip()
        
        # Check if we need separator
        if old_text and new_text:
            # If old text doesn't end with punctuation and new doesn't start with punctuation
            if not old_text[-1] in '.!?:;,)]\n' and not new_text[0] in '.!?:;,([':
                self.full_text = old_text + " " + new_text
            else:
                self.full_text = old_text + new_text
        else:
            self.full_text = old_text + new_text
        
        # Update page range
        if page_number not in self.page_range:
            self.page_range.append(page_number)
            self.page_range.sort()
        
        logger.debug(f"Merged text into {self.id}, now spans pages {self.page_range}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "full_text": self.full_text,
            "page_range": self.page_range,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }


class CrossPageMerger:
    """
    Handles merging of nodes that span multiple pages.
    
    The key insight is that VLM processes pages independently, so when
    an Article continues on the next page, we get two partial nodes
    with the same ID. This class merges them correctly.
    
    Usage:
        ```python
        merger = CrossPageMerger()
        
        # Process each page
        for page_num, page_result in enumerate(vlm_results, 1):
            merger.add_page_nodes(page_result.nodes, page_num)
        
        # Get merged results
        all_nodes = merger.get_merged_nodes()
        ```
    """
    
    def __init__(self):
        """Initialize the merger."""
        self.nodes_by_id: Dict[str, MergeableNode] = {}
        self.relations: List[Dict[str, Any]] = []
        self.page_count = 0
        
        # Track context for cross-page continuation
        self.current_context: Dict[str, Any] = {}
    
    def add_page_nodes(
        self,
        nodes: List[Dict[str, Any]],
        page_number: int,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add nodes from a single page, merging with existing nodes.
        
        This is the CRITICAL function that fixes the lost content problem.
        
        Logic:
        1. For each new node, check if ID already exists
        2. If exists: APPEND text instead of replacing
        3. If new: Add to collection
        4. Update page ranges
        
        Args:
            nodes: List of node dicts from VLM
            page_number: Current page number
            context: Optional context from VLM (current_article, pending_text, etc.)
        """
        self.page_count = max(self.page_count, page_number)
        
        if context:
            self.current_context = context
        
        for node_data in nodes:
            node_id = node_data.get("id", "")
            if not node_id:
                logger.warning(f"Node without ID on page {page_number}: {node_data}")
                continue
            
            if node_id in self.nodes_by_id:
                # EXISTING NODE: Merge text (DON'T REPLACE!)
                existing_node = self.nodes_by_id[node_id]
                new_text = node_data.get("full_text", "")
                
                logger.info(
                    f"Merging continuation of '{node_id}' from page {page_number} "
                    f"(existing: {len(existing_node.full_text)} chars, "
                    f"new: {len(new_text)} chars)"
                )
                
                existing_node.merge_text(new_text, page_number)
                
                # Update title if the new one is more complete
                new_title = node_data.get("title", "")
                if new_title and len(new_title) > len(existing_node.title):
                    existing_node.title = new_title
                
            else:
                # NEW NODE: Create and add
                new_node = MergeableNode(
                    id=node_id,
                    type=node_data.get("type", "Unknown"),
                    title=node_data.get("title", ""),
                    full_text=node_data.get("full_text", ""),
                    page_range=[page_number],
                    parent_id=node_data.get("parent_id"),
                    metadata={
                        "source_page": page_number,
                        **node_data.get("metadata", {})
                    }
                )
                self.nodes_by_id[node_id] = new_node
                
                logger.info(f"Added new node '{node_id}' from page {page_number}")
    
    def add_relations(self, relations: List[Dict[str, Any]]) -> None:
        """
        Add relations, avoiding duplicates.
        
        Args:
            relations: List of relation dicts
        """
        for rel in relations:
            rel_key = (rel.get("source"), rel.get("target"), rel.get("type"))
            
            # Check for duplicate
            is_duplicate = any(
                (r.get("source"), r.get("target"), r.get("type")) == rel_key
                for r in self.relations
            )
            
            if not is_duplicate:
                self.relations.append(rel)
    
    def handle_pending_text(
        self,
        pending_text: Optional[str],
        pending_node_id: Optional[str],
        page_number: int
    ) -> None:
        """
        Handle text that was pending from previous page.
        
        VLM may return pending_text when content is cut mid-sentence.
        This text should be prepended to the next occurrence of the node.
        
        Args:
            pending_text: Text pending from previous page
            pending_node_id: ID of node this text belongs to
            page_number: Current page number
        """
        if not pending_text or not pending_node_id:
            return
        
        if pending_node_id in self.nodes_by_id:
            node = self.nodes_by_id[pending_node_id]
            
            # Check if pending text is at the start (from previous page)
            if pending_text.strip() not in node.full_text:
                # Prepend or append based on context
                logger.info(
                    f"Handling pending text for '{pending_node_id}' "
                    f"({len(pending_text)} chars)"
                )
                node.merge_text(pending_text, page_number)
    
    def get_merged_nodes(self) -> List[MergeableNode]:
        """
        Get all merged nodes.
        
        Returns:
            List of merged nodes with complete content
        """
        return list(self.nodes_by_id.values())
    
    def get_merged_relations(self) -> List[Dict[str, Any]]:
        """
        Get all relations.
        
        Returns:
            List of unique relations
        """
        return self.relations
    
    def get_nodes_by_type(self, node_type: str) -> List[MergeableNode]:
        """
        Get nodes filtered by type.
        
        Args:
            node_type: Type to filter by (e.g., "Article", "Chapter")
            
        Returns:
            List of matching nodes
        """
        return [
            node for node in self.nodes_by_id.values()
            if node.type == node_type
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of merged results.
        
        Returns:
            Summary statistics
        """
        nodes = list(self.nodes_by_id.values())
        
        type_counts = {}
        for node in nodes:
            type_counts[node.type] = type_counts.get(node.type, 0) + 1
        
        cross_page_nodes = [n for n in nodes if len(n.page_range) > 1]
        
        return {
            "total_nodes": len(nodes),
            "total_relations": len(self.relations),
            "page_count": self.page_count,
            "nodes_by_type": type_counts,
            "cross_page_nodes": len(cross_page_nodes),
            "cross_page_details": [
                {"id": n.id, "type": n.type, "pages": n.page_range}
                for n in cross_page_nodes
            ]
        }


def merge_page_results(
    all_page_results: List[Dict[str, Any]],
    include_relations: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Merge results from multiple pages into unified structure.
    
    This is the main entry point for the merge functionality.
    
    Args:
        all_page_results: List of dicts, each containing:
            - page_number: int
            - nodes: List[Dict]
            - relations: List[Dict]
            - context: Dict (optional, with pending_text, pending_node_id, etc.)
        include_relations: Whether to include relations in output
        
    Returns:
        Tuple of:
            - merged_nodes: List[Dict] - All nodes with merged content
            - merged_relations: List[Dict] - All unique relations
            - summary: Dict - Merge statistics
            
    Example:
        >>> page_results = [
        ...     {
        ...         "page_number": 1,
        ...         "nodes": [{"id": "dieu_1", "type": "Article", "full_text": "Điều 1. Phần đầu..."}],
        ...         "relations": [],
        ...         "context": {"pending_text": None}
        ...     },
        ...     {
        ...         "page_number": 2,
        ...         "nodes": [{"id": "dieu_1", "type": "Article", "full_text": "...phần tiếp theo của Điều 1"}],
        ...         "relations": [],
        ...         "context": {}
        ...     }
        ... ]
        >>> nodes, relations, summary = merge_page_results(page_results)
        >>> print(len(nodes))  # 1 (merged)
        >>> print(nodes[0]["full_text"])  # Contains both parts
    """
    merger = CrossPageMerger()
    
    for page_result in all_page_results:
        page_num = page_result.get("page_number", 0)
        nodes = page_result.get("nodes", [])
        relations = page_result.get("relations", [])
        context = page_result.get("context", {})
        
        # Handle pending text from previous page
        if context:
            merger.handle_pending_text(
                context.get("pending_text"),
                context.get("pending_node_id"),
                page_num
            )
        
        # Add nodes with merging
        merger.add_page_nodes(nodes, page_num, context)
        
        # Add relations
        if include_relations:
            merger.add_relations(relations)
    
    # Get results
    merged_nodes = [node.to_dict() for node in merger.get_merged_nodes()]
    merged_relations = merger.get_merged_relations()
    summary = merger.get_summary()
    
    logger.info(
        f"Merge complete: {summary['total_nodes']} nodes "
        f"({summary['cross_page_nodes']} cross-page), "
        f"{summary['total_relations']} relations"
    )
    
    return merged_nodes, merged_relations, summary


# =============================================================================
# NEW: Function to work directly with StructureNode objects
# =============================================================================

def merge_nodes_into_dict(
    existing_nodes: Dict[str, Any],
    new_nodes: List[Any],
    page_number: int,
    logger_instance: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Merge new page nodes into existing node dictionary.
    
    This function works directly with StructureNode objects from hybrid_extractor.
    For same IDs, it APPENDs text instead of replacing (fixing the critical bug).
    
    Args:
        existing_nodes: Dict mapping node_id -> StructureNode (any object with
                        id, full_text, page_range attributes)
        new_nodes: List of StructureNode objects from current page
        page_number: Current page number
        logger_instance: Optional logger for debug output
        
    Returns:
        Updated dict with merged nodes
        
    Example:
        >>> merged = {}
        >>> for page_num, page_nodes in enumerate(all_pages, 1):
        ...     merged = merge_nodes_into_dict(merged, page_nodes, page_num)
    """
    log = logger_instance or logger
    
    for node in new_nodes:
        node_id = node.id
        
        if node_id in existing_nodes:
            # EXISTING NODE: Merge text (CRITICAL FIX)
            existing = existing_nodes[node_id]
            new_text = node.full_text or ""
            
            # Check for duplicate text
            if new_text.strip() and new_text.strip() not in existing.full_text:
                # Smart merge
                old_text = existing.full_text.rstrip()
                clean_new = new_text.lstrip()
                
                if old_text and clean_new:
                    # Check if we need separator
                    if not old_text[-1] in '.!?:;,)]\n' and not clean_new[0] in '.!?:;,([':
                        existing.full_text = old_text + " " + clean_new
                    else:
                        existing.full_text = old_text + clean_new
                else:
                    existing.full_text = (old_text or "") + (clean_new or "")
                
                log.info(
                    f"Merged '{node_id}' from page {page_number}: "
                    f"+{len(new_text)} chars -> total {len(existing.full_text)} chars"
                )
            
            # Update page range
            if page_number not in existing.page_range:
                existing.page_range.append(page_number)
                existing.page_range.sort()
            
            # Update title if new is more complete
            if node.title and len(node.title) > len(existing.title):
                existing.title = node.title
                
        else:
            # NEW NODE: Add to dict
            # Ensure page_range includes current page
            if page_number not in node.page_range:
                node.page_range.append(page_number)
            
            existing_nodes[node_id] = node
            log.debug(f"Added new node '{node_id}' from page {page_number}")
    
    return existing_nodes


# Alias for backward compatibility
merge_page_results_simple = merge_nodes_into_dict


# Test function
def _test_merge():
    """Test the merge functionality."""
    print("Testing cross-page merge...")
    
    # Simulate 2 pages with Article 1 split across them
    page_results = [
        {
            "page_number": 1,
            "nodes": [
                {
                    "id": "quyet_dinh",
                    "type": "Document",
                    "title": "QUYẾT ĐỊNH",
                    "full_text": "Căn cứ..."
                },
                {
                    "id": "dieu_1",
                    "type": "Article",
                    "title": "Điều 1",
                    "full_text": "Điều 1. Cập nhật Quy chế đào tạo theo học chế tín chỉ. Phần đầu của điều này..."
                }
            ],
            "relations": [
                {"source": "quyet_dinh", "target": "dieu_1", "type": "CONTAINS"}
            ],
            "context": {
                "current_article_id": "dieu_1",
                "pending_text": None
            }
        },
        {
            "page_number": 2,
            "nodes": [
                {
                    "id": "dieu_1",  # Same ID - should be MERGED
                    "type": "Article",
                    "title": "Điều 1",
                    "full_text": "...tiếp tục nội dung Điều 1 từ trang trước. Đây là phần cuối."
                },
                {
                    "id": "dieu_2",
                    "type": "Article",
                    "title": "Điều 2",
                    "full_text": "Điều 2. Nội dung mới."
                }
            ],
            "relations": [
                {"source": "quyet_dinh", "target": "dieu_1", "type": "CONTAINS"},  # Duplicate
                {"source": "quyet_dinh", "target": "dieu_2", "type": "CONTAINS"}
            ],
            "context": {}
        }
    ]
    
    nodes, relations, summary = merge_page_results(page_results)
    
    print(f"\nMerge Summary:")
    print(f"  Total nodes: {summary['total_nodes']}")
    print(f"  Cross-page nodes: {summary['cross_page_nodes']}")
    print(f"  Relations: {summary['total_relations']}")
    
    print(f"\nMerged Nodes:")
    for node in nodes:
        print(f"  - {node['id']} ({node['type']})")
        print(f"    Pages: {node['page_range']}")
        print(f"    Text length: {len(node['full_text'])} chars")
        if len(node['page_range']) > 1:
            print(f"    ✓ Successfully merged across pages!")
    
    print(f"\nRelations (deduplicated): {len(relations)}")
    for rel in relations:
        print(f"  - {rel['source']} -> {rel['target']} ({rel['type']})")
    
    # Verify merge worked
    dieu_1 = next((n for n in nodes if n["id"] == "dieu_1"), None)
    if dieu_1:
        if len(dieu_1["page_range"]) == 2 and "Phần đầu" in dieu_1["full_text"] and "phần cuối" in dieu_1["full_text"]:
            print("\n✓ TEST PASSED: Điều 1 correctly merged from 2 pages!")
        else:
            print("\n✗ TEST FAILED: Điều 1 not properly merged")
    else:
        print("\n✗ TEST FAILED: Điều 1 not found")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    _test_merge()


# =============================================================================
# POST-PROCESSING: Auto-fix missing relations for amendment documents
# =============================================================================

def auto_fix_amendment_relations(
    structure_data: Dict[str, Any],
    logger_instance: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Automatically fix missing CONTAINS relations for amendment documents.
    
    For amendment documents (văn bản sửa đổi), clauses like "Khoản 3 Điều 4"
    often don't have proper parent relations. This function:
    1. Identifies orphan clauses (no parent relation)
    2. Links them to the main Article (usually Điều 1)
    3. Adds metadata about what they modify
    
    Args:
        structure_data: Dict with articles, clauses, relations keys
        logger_instance: Optional logger
        
    Returns:
        Updated structure_data with fixed relations
    """
    log = logger_instance or logger
    
    articles = structure_data.get("articles", [])
    clauses = structure_data.get("clauses", [])
    relations = structure_data.get("relations", [])
    document = structure_data.get("document", {})
    
    # Check if this is an amendment document
    doc_title = document.get("title", "") if document else ""
    is_amendment = any(kw in doc_title.lower() for kw in 
                       ["sửa đổi", "bổ sung", "cập nhật", "thay thế"])
    
    if not is_amendment:
        log.debug("Not an amendment document, skipping auto-fix")
        return structure_data
    
    log.info(f"Detected amendment document: '{doc_title[:50]}...'")
    
    # Build set of existing relations (target -> source)
    existing_targets = set()
    for rel in relations:
        if rel.get("type") == "CONTAINS":
            existing_targets.add(rel["target"])
    
    # Find main amendment article (usually Điều 1 or first article mentioning modifications)
    main_article_id = None
    for article in articles:
        article_text = article.get("full_text", "").lower()
        if any(kw in article_text for kw in ["cập nhật", "sửa đổi", "bổ sung", "như sau"]):
            main_article_id = article.get("id")
            break
    
    if not main_article_id and articles:
        # Fallback to first article
        main_article_id = articles[0].get("id")
    
    if not main_article_id:
        log.warning("No main article found for amendment relations")
        return structure_data
    
    log.info(f"Using '{main_article_id}' as main amendment article")
    
    # Find orphan clauses and create relations
    added_relations = 0
    for clause in clauses:
        clause_id = clause.get("id", "")
        clause_title = clause.get("title", "")
        
        # Skip if already has relation or is not a modification clause
        if clause_id in existing_targets:
            continue
        
        # Check if this looks like a modification clause (Khoản X Điều Y)
        import re
        is_modification_clause = bool(
            re.search(r'(khoản|mục|điểm)\s*\d+.*điều\s*\d+', clause_title.lower()) or
            re.search(r'(khoản|mục|điểm)\s*\w+.*điều\s*\d+', clause_id.lower())
        )
        
        if is_modification_clause or clause_id not in existing_targets:
            # Create CONTAINS relation from main article
            new_relation = {
                "source": main_article_id,
                "target": clause_id,
                "type": "CONTAINS"
            }
            relations.append(new_relation)
            existing_targets.add(clause_id)
            added_relations += 1
            
            log.info(f"Added relation: {main_article_id} -> {clause_id}")
            
            # Also try to extract modification metadata
            match = re.search(r'(khoản\s*\d+)?\s*(điều\s*\d+)', clause_title.lower())
            if match:
                clause["metadata"] = clause.get("metadata", {})
                clause["metadata"]["modifies"] = match.group(0).title()
    
    log.info(f"Auto-fixed {added_relations} missing relations for amendment document")
    
    structure_data["relations"] = relations
    return structure_data
