# RAG Indexing Guide - VLM JSON to Weaviate

## Overview
Hệ thống indexing mới sử dụng JSON output từ VLM (Vision Language Model) để index vào Weaviate vector database.

## Workflow

```
PDF → VLM Extraction → JSON → index_semantic_data.py → Weaviate
```

### Step 1: Extract với VLM
Sử dụng VLM để extract cấu trúc từ PDF:

```bash
python scripts/run_full_extraction.py --input data/your_document.pdf
```

Output: `data/extraction_results/full_extraction_YYYYMMDD_HHMMSS.json`

### Step 2: Index vào Weaviate

```bash
# Đảm bảo Weaviate đang chạy
docker-compose up -d weaviate

# Index JSON vào Weaviate
python scripts/index_semantic_data.py \
    --input data/extraction_results/full_extraction_20251205_151556.json \
    --doc-type regulation \
    --batch-size 50
```

### Step 3: Verify Indexing

```bash
# Kiểm tra Weaviate có dữ liệu
python check_weaviate.py

# Test search
python test_quick.py
```

## JSON Structure Required

```json
{
  "stage1_structure": {
    "articles": [
      {
        "id": "dieu_1",
        "number": 1,
        "title": "Điều 1. Title",
        "full_text": "Content...",
        "page": 1,
        "chapter": "Chương 1",
        "clauses": [
          {
            "id": "khoan_1_dieu_1",
            "title": "Khoản 1",
            "full_text": "Clause content..."
          }
        ]
      }
    ],
    "clauses": [
      {
        "id": "khoan_3_dieu_4",
        "title": "Khoản 3 Điều 4",
        "full_text": "Standalone clause content...",
        "page": 2
      }
    ]
  },
  "relations": [
    {
      "source_id": "khoan_1_dieu_1",
      "target_id": "dieu_10",
      "relation_type": "references",
      "evidence": "Context explanation"
    }
  ],
  "metadata": {
    "source_file": "document.pdf",
    "doc_type": "regulation",
    "extraction_date": "2024-12-06"
  }
}
```

## Indexing Strategy

### Chunking
- Mỗi **Khoản (Clause)** hoặc **Điều (Article)** = 1 Document Chunk
- Content = Title + Clause Content + Evidence (từ relations)

### Metadata
- `source_id`: Map với Neo4j KG (e.g., "khoan_1_dieu_23")
- `article_number`: Số hiệu điều
- `doc_type`: Loại văn bản (regulation, policy)
- `structure_type`: "clause" hoặc "article"
- `page`, `chapter`: Vị trí trong văn bản

### Embedding Model
- Model: `intfloat/multilingual-e5-base`
- Dimensions: 768
- Distance metric: COSINE

## Weaviate Configuration

### Collection Schema
- Name: `VietnameseDocumentV3`
- Vector dimensions: 768
- Distance metric: COSINE
- Properties: 19 fields (text, doc_id, chunk_id, title, article_number, etc.)

### Recreate Collection
Nếu cần recreate schema:

```bash
python scripts/recreate_weaviate_collection.py
```

## Utility Scripts

- `check_weaviate.py` - Kiểm tra dữ liệu trong Weaviate
- `clear_weaviate.py` - Xóa toàn bộ dữ liệu
- `inspect_vectors.py` - Debug vector storage format
- `test_quick.py` - Test RAG retrieval nhanh
- `test_rag_search.py` - Test comprehensive search

## Troubleshooting

### Issue: Vectors stored with wrong dimensions
**Solution**: Upgrade weaviate-client
```bash
pip install --upgrade weaviate-client==4.18.3
```

### Issue: Clauses not indexed
**Solution**: Check variable name collision in `index_semantic_data.py` line 110
```python
article_clauses = article.get("clauses", [])  # NOT: clauses = ...
```

### Issue: Schema property mismatch
**Solution**: Ensure return_props matches actual schema in `weaviate_vector_adapter.py`
```python
return_props = ["text", "doc_id", "chunk_id", "title", "page", 
                "doc_type", "faculty", "year", "subject", "section", 
                "subsection", "language", "structure_type", "chapter",
                "article", "article_number", "filename", "metadata_json"]
```

## Current Data

- **Indexed**: 6 documents
  - 2 articles: Điều 1, Điều 2
  - 4 clauses: Khoản 3 Điều 4, Khoản 2 Điều 12, Khoản 3 Điều 14, Khoản 1 Điều 23
- **Source**: `full_extraction_20251205_151556.json`
- **Vector DB**: Weaviate @ http://localhost:8090
- **Collection**: VietnameseDocumentV3 (768D vectors, COSINE distance)

## Integration with LLM

RAG context được sử dụng bởi Orchestrator:
- Smart Planner quyết định khi nào dùng RAG
- Vector search retrieve documents
- LLM sử dụng retrieved context để trả lời

Verified working ✅
