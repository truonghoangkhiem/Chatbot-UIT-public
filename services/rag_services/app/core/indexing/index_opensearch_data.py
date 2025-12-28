#!/usr/bin/env python3
"""
Script để indexing dữ liệu từ VLM JSON output vào OpenSearch (BM25 keyword search).

Chiến lược:
- Chunk theo đơn vị logic: Mỗi Khoản (Clause) hoặc Điều (Article) là một Document
- Content = Article Title + Clause Content + Evidence (nếu có)
- Metadata giàu: source_id (map với Neo4j), article_number, doc_type, page_number

Usage:
    python index_opensearch_data.py --input data/full_extraction_quyche.json --doc-type regulation
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add parent directories to path
current_dir = Path(__file__).parent.absolute()
rag_services_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(rag_services_dir))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ChunkData:
    """Dữ liệu cho một chunk (Khoản hoặc Điều)"""
    source_id: str  # ID khớp với Neo4j (e.g., "khoan_1_dieu_23")
    content: str  # Nội dung đầy đủ để indexing
    article_number: Optional[int]  # Số hiệu điều
    article_title: str  # Tiêu đề điều
    doc_type: str  # Loại văn bản
    page_number: Optional[int]  # Trang xuất hiện
    chapter: Optional[str]  # Chương (nếu có)
    structure_type: str  # "clause" hoặc "article"
    clause_index: Optional[int]  # Thứ tự khoản trong điều


def load_and_process_json(file_path: str, doc_type: str = "regulation") -> List[ChunkData]:
    """
    Load JSON file và flatten thành danh sách chunks.
    
    Args:
        file_path: Đường dẫn đến file JSON
        doc_type: Loại văn bản (regulation, policy, etc.)
        
    Returns:
        Danh sách ChunkData
    """
    import re
    
    logger.info(f"Loading JSON from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = []
    
    # Build evidence map từ relations (nếu có)
    evidence_map = {}
    if "relations" in data:
        for rel in data["relations"]:
            source_id = rel.get("source_article_id") or rel.get("source_id")
            if source_id and "evidence" in rel:
                if source_id not in evidence_map:
                    evidence_map[source_id] = []
                evidence_map[source_id].append(rel["evidence"])
    
    # Lấy filename từ file_path
    filename = Path(file_path).stem
    
    # Process stage1_structure
    if "stage1_structure" not in data:
        logger.warning("No 'stage1_structure' found in JSON!")
        return chunks
    
    structure = data["stage1_structure"]
    articles = structure.get("articles", [])
    clauses = structure.get("clauses", [])
    
    logger.info(f"Found {len(articles)} articles and {len(clauses)} clauses in structure")
    
    # Process articles
    for article in articles:
        article_id = article.get("id", "")
        article_title = article.get("title", "")
        article_number = article.get("number")  # Số hiệu điều
        page_number = article.get("page")
        chapter = article.get("chapter")
        
        # Extract article number từ title nếu không có field 'number'
        if article_number is None and article_title:
            match = re.search(r'Điều\s+(\d+)', article_title)
            if match:
                article_number = int(match.group(1))
        
        article_clauses = article.get("clauses", [])
        
        if article_clauses:
            # Có khoản -> mỗi khoản là một chunk
            for idx, clause in enumerate(article_clauses, start=1):
                clause_id = clause.get("id", f"khoan_{idx}_{article_id}")
                clause_title = clause.get("title", "")
                clause_content = clause.get("full_text", "")
                
                # Build full content for indexing
                content_parts = [article_title]
                if clause_title and clause_title != article_title:
                    content_parts.append(clause_title)
                if clause_content:
                    content_parts.append(clause_content)
                
                # Add evidence nếu có
                if clause_id in evidence_map:
                    evidence_text = " | ".join(evidence_map[clause_id])
                    content_parts.append(f"[Ngữ cảnh: {evidence_text}]")
                
                full_content = "\n".join(content_parts)
                
                chunk = ChunkData(
                    source_id=clause_id,
                    content=full_content,
                    article_number=article_number,
                    article_title=article_title,
                    doc_type=doc_type,
                    page_number=page_number,
                    chapter=chapter,
                    structure_type="clause",
                    clause_index=idx
                )
                chunks.append(chunk)
                
        else:
            # Không có khoản -> toàn bộ điều là một chunk
            article_content = article.get("full_text", "")
            
            content_parts = [article_title]
            if article_content:
                content_parts.append(article_content)
            
            # Add evidence
            if article_id in evidence_map:
                evidence_text = " | ".join(evidence_map[article_id])
                content_parts.append(f"[Ngữ cảnh: {evidence_text}]")
            
            full_content = "\n".join(content_parts)
            
            chunk = ChunkData(
                source_id=article_id,
                content=full_content,
                article_number=article_number,
                article_title=article_title,
                doc_type=doc_type,
                page_number=page_number,
                chapter=chapter,
                structure_type="article",
                clause_index=None
            )
            chunks.append(chunk)
    
    # Process standalone clauses (clauses ở cùng cấp với articles)
    logger.info(f"Processing {len(clauses)} standalone clauses...")
    for clause in clauses:
        clause_id = clause.get("id", "")
        clause_title = clause.get("title", "")
        clause_content = clause.get("full_text", "")
        page_number = clause.get("page")
        chapter = clause.get("chapter")
        
        # Extract article number from clause title if possible
        article_number = None
        match = re.search(r'Điều\s+(\d+)', clause_title)
        if match:
            article_number = int(match.group(1))
        
        # Build full content
        content_parts = []
        if clause_title:
            content_parts.append(clause_title)
        if clause_content:
            content_parts.append(clause_content)
        
        # Add evidence if exists
        if clause_id in evidence_map:
            evidence_text = " | ".join(evidence_map[clause_id])
            content_parts.append(f"[Ngữ cảnh: {evidence_text}]")
        
        full_content = "\n".join(content_parts)
        
        chunk = ChunkData(
            source_id=clause_id,
            content=full_content,
            article_number=article_number,
            article_title=clause_title,
            doc_type=doc_type,
            page_number=page_number,
            chapter=chapter,
            structure_type="clause",
            clause_index=None
        )
        chunks.append(chunk)
    
    logger.info(f"Processed {len(chunks)} chunks from JSON")
    return chunks


def convert_to_opensearch_documents(chunks_data: List[ChunkData], filename: str) -> List[Dict[str, Any]]:
    """
    Convert ChunkData thành OpenSearch documents.
    
    Args:
        chunks_data: Danh sách ChunkData
        filename: Tên file gốc
        
    Returns:
        Danh sách documents cho OpenSearch
    """
    documents = []
    
    for idx, chunk_data in enumerate(chunks_data):
        # Build metadata
        metadata = {
            "source_id": chunk_data.source_id,  # QUAN TRỌNG: Map với Neo4j
            "article_number": chunk_data.article_number,
            "article_title": chunk_data.article_title,
            "structure_type": chunk_data.structure_type,
            "filename": filename,
        }
        
        if chunk_data.chapter:
            metadata["chapter"] = chunk_data.chapter
            metadata["section"] = chunk_data.chapter
        
        if chunk_data.clause_index is not None:
            metadata["clause_index"] = chunk_data.clause_index
        
        if chunk_data.page_number is not None:
            metadata["page"] = chunk_data.page_number
        
        document = {
            "doc_id": filename,  # Document ID chung
            "chunk_id": chunk_data.source_id,  # Chunk ID unique
            "text": chunk_data.content,
            "title": chunk_data.article_title,
            "doc_type": chunk_data.doc_type,
            "faculty": None,  # Không áp dụng cho regulation
            "year": None,
            "subject": None,
            "language": "vi",
            "metadata": metadata,
            "char_spans": []  # Sẽ được tự động extract bởi OpenSearch client
        }
        
        documents.append(document)
    
    return documents


def index_to_opensearch(documents: List[Dict[str, Any]], batch_size: int = 100) -> tuple:
    """
    Index documents vào OpenSearch.
    
    Args:
        documents: Danh sách documents cần index
        batch_size: Số lượng documents mỗi batch
        
    Returns:
        Tuple (success_count, failed_count)
    """
    from infrastructure.store.opensearch.client import OpenSearchClient
    
    try:
        client = OpenSearchClient()
        logger.info("✓ OpenSearch client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize OpenSearch client: {e}")
        return 0, len(documents)
    
    total_success = 0
    total_failed = 0
    
    # Index theo batches
    total = len(documents)
    total_batches = (total + batch_size - 1) // batch_size
    
    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
        
        try:
            success, failed = client.bulk_index_documents(batch)
            total_success += success
            total_failed += failed
            logger.info(f"✓ Batch {batch_num}: {success} success, {failed} failed")
        except Exception as e:
            logger.error(f"✗ Batch {batch_num} failed: {e}")
            total_failed += len(batch)
    
    return total_success, total_failed


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Index VLM JSON output vào OpenSearch (BM25 keyword search)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSON file (e.g., data/full_extraction_quyche.json)"
    )
    parser.add_argument(
        "--doc-type",
        default="regulation",
        help="Document type (regulation, policy, etc.)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for indexing (default: 100)"
    )
    parser.add_argument(
        "--clear-index",
        action="store_true",
        help="Clear existing index before indexing"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
    
    filename = Path(args.input).stem
    
    # Optional: Clear existing index
    if args.clear_index:
        logger.info("=" * 80)
        logger.info("CLEARING EXISTING INDEX")
        logger.info("=" * 80)
        try:
            from infrastructure.store.opensearch.client import OpenSearchClient
            client = OpenSearchClient()
            # Delete documents with matching doc_id
            delete_result = client.client.delete_by_query(
                index=client.index_name,
                body={
                    "query": {
                        "term": {"doc_id": filename}
                    }
                }
            )
            deleted = delete_result.get("deleted", 0)
            logger.info(f"Deleted {deleted} existing documents with doc_id={filename}")
        except Exception as e:
            logger.warning(f"Could not clear index: {e}")
    
    # Step 1: Load and process JSON
    logger.info("=" * 80)
    logger.info("STEP 1: Loading and processing JSON")
    logger.info("=" * 80)
    
    chunks_data = load_and_process_json(args.input, args.doc_type)
    
    if not chunks_data:
        logger.error("No chunks extracted from JSON!")
        return
    
    logger.info(f"Extracted {len(chunks_data)} chunks")
    
    # Step 2: Convert to OpenSearch documents
    logger.info("=" * 80)
    logger.info("STEP 2: Converting to OpenSearch documents")
    logger.info("=" * 80)
    
    documents = convert_to_opensearch_documents(chunks_data, filename)
    logger.info(f"Created {len(documents)} OpenSearch documents")
    
    # Preview first document
    if documents:
        logger.info("Preview first document:")
        preview = documents[0]
        logger.info(f"  doc_id: {preview['doc_id']}")
        logger.info(f"  chunk_id: {preview['chunk_id']}")
        logger.info(f"  title: {preview['title'][:80]}..." if len(preview['title']) > 80 else f"  title: {preview['title']}")
        logger.info(f"  doc_type: {preview['doc_type']}")
        logger.info(f"  text length: {len(preview['text'])} chars")
    
    # Step 3: Index to OpenSearch
    logger.info("=" * 80)
    logger.info("STEP 3: Indexing to OpenSearch")
    logger.info("=" * 80)
    
    success_count, failed_count = index_to_opensearch(documents, args.batch_size)
    
    # Summary
    logger.info("=" * 80)
    logger.info("INDEXING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total documents: {len(documents)}")
    logger.info(f"Successfully indexed: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Source file: {args.input}")
    logger.info(f"Document type: {args.doc_type}")
    
    if failed_count == 0:
        logger.info("✓ INDEXING COMPLETED SUCCESSFULLY!")
    else:
        logger.warning(f"⚠ Completed with {failed_count} failures")


if __name__ == "__main__":
    main()
