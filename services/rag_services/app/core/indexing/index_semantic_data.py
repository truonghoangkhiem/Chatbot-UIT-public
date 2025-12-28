#!/usr/bin/env python3
"""
Script để indexing dữ liệu từ VLM JSON output vào Vector Database.

Chiến lược:
- Chunk theo đơn vị logic: Mỗi Khoản (Clause) hoặc Điều (Article) là một Document
- Content = Article Title + Clause Content + Evidence (nếu có)
- Metadata giàu: source_id (map với Neo4j), article_number, doc_type, page_number

Usage:
    python index_semantic_data.py --input data/full_extraction_quyche.json --doc-type regulation
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
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.domain.models import DocumentChunk, DocumentMetadata, DocumentLanguage
from adapters.weaviate_vector_adapter import WeaviateVectorAdapter
from infrastructure.store.vector.weaviate_store import get_weaviate_client, DOCUMENT_COLLECTION

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
    content: str  # Nội dung đầy đủ để embedding
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
            # Ví dụ: "Điều 23. Tổ chức đào tạo" -> 23
            import re
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
                
                # Build full content for embedding
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
        # Ví dụ: "Khoản 3 Điều 4" -> article_number = 4
        article_number = None
        import re
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
            article_title=clause_title,  # Use clause title as article title
            doc_type=doc_type,
            page_number=page_number,
            chapter=chapter,
            structure_type="clause",
            clause_index=None
        )
        chunks.append(chunk)
    
    logger.info(f"Processed {len(chunks)} chunks from JSON")
    return chunks


def convert_to_document_chunks(chunks_data: List[ChunkData], filename: str) -> List[DocumentChunk]:
    """
    Convert ChunkData thành DocumentChunk domain models.
    
    Args:
        chunks_data: Danh sách ChunkData
        filename: Tên file gốc
        
    Returns:
        Danh sách DocumentChunk
    """
    document_chunks = []
    
    for idx, chunk_data in enumerate(chunks_data):
        # Build metadata với các field quan trọng
        extra = {
            "source_id": chunk_data.source_id,  # QUAN TRỌNG: Map với Neo4j
            "article_number": chunk_data.article_number,
            "article_title": chunk_data.article_title,
            "structure_type": chunk_data.structure_type,
            "filename": filename
        }
        
        if chunk_data.chapter:
            extra["chapter"] = chunk_data.chapter
        
        if chunk_data.clause_index is not None:
            extra["clause_index"] = chunk_data.clause_index
        
        metadata = DocumentMetadata(
            doc_id=filename,  # Document ID chung
            chunk_id=chunk_data.source_id,  # Chunk ID unique
            title=chunk_data.article_title,
            page=chunk_data.page_number,
            doc_type=chunk_data.doc_type,
            faculty=None,  # Không áp dụng cho regulation
            year=None,
            subject=None,
            section=chunk_data.chapter,  # Dùng chapter làm section
            subsection=None,
            language=DocumentLanguage.VIETNAMESE,
            extra=extra
        )
        
        document_chunk = DocumentChunk(
            text=chunk_data.content,
            metadata=metadata,
            chunk_index=idx,
            char_spans=None
        )
        
        document_chunks.append(document_chunk)
    
    return document_chunks


async def index_batches(
    vector_adapter: WeaviateVectorAdapter,
    chunks: List[DocumentChunk],
    batch_size: int = 50
):
    """
    Index chunks theo batch để tránh memory issues.
    
    Args:
        vector_adapter: Weaviate adapter instance
        chunks: Danh sách chunks cần index
        batch_size: Số lượng chunks mỗi batch
    """
    total = len(chunks)
    logger.info(f"Starting indexing {total} chunks in batches of {batch_size}")
    
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)")
        
        success = await vector_adapter.index_documents(batch)
        
        if success:
            logger.info(f"✓ Batch {batch_num} indexed successfully")
        else:
            logger.error(f"✗ Batch {batch_num} failed!")
    
    logger.info(f"Completed indexing all {total} chunks")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Index VLM JSON output vào Vector Database"
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
        default=50,
        help="Batch size for indexing (default: 50)"
    )
    parser.add_argument(
        "--weaviate-url",
        default="http://localhost:8090",
        help="Weaviate server URL"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
    
    filename = Path(args.input).stem
    
    # Step 1: Load and process JSON
    logger.info("=" * 80)
    logger.info("STEP 1: Loading and processing JSON")
    logger.info("=" * 80)
    
    chunks_data = load_and_process_json(args.input, args.doc_type)
    
    if not chunks_data:
        logger.error("No chunks extracted from JSON!")
        return
    
    logger.info(f"Extracted {len(chunks_data)} chunks")
    
    # Step 2: Convert to DocumentChunk models
    logger.info("=" * 80)
    logger.info("STEP 2: Converting to DocumentChunk models")
    logger.info("=" * 80)
    
    document_chunks = convert_to_document_chunks(chunks_data, filename)
    logger.info(f"Created {len(document_chunks)} DocumentChunk objects")
    
    # Step 3: Initialize Vector Adapter
    logger.info("=" * 80)
    logger.info("STEP 3: Initializing Vector Database connection")
    logger.info("=" * 80)
    
    try:
        # Import embedding model
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from app.config.settings import settings
        
        embedding_model = HuggingFaceEmbedding(model_name=settings.emb_model)
        logger.info(f"✓ Embedding model loaded: {settings.emb_model}")
        
        vector_adapter = WeaviateVectorAdapter(
            weaviate_url=args.weaviate_url,
            embedding_model=embedding_model,
            api_key=None
        )
        logger.info("✓ Weaviate adapter initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize vector adapter: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Index documents
    logger.info("=" * 80)
    logger.info("STEP 4: Indexing documents to Vector Database")
    logger.info("=" * 80)
    
    try:
        await index_batches(vector_adapter, document_chunks, args.batch_size)
        
        logger.info("=" * 80)
        logger.info("✓ INDEXING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Total chunks indexed: {len(document_chunks)}")
        logger.info(f"Source file: {args.input}")
        logger.info(f"Document type: {args.doc_type}")
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
