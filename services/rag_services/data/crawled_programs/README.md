# Crawled Program Data

This folder contains curriculum data crawled from the UIT official student portal.

## Data Source

- **Website**: https://student.uit.edu.vn
- **Type**: Training curriculum programs
- **Language**: Vietnamese

## Available Data

### 1. Khoa học Máy tính (Computer Science) - Khóa 19 (2024)

**File**: `cu-nhan-nganh-khoa-hoc-may-tinh-ap-dung-tu-khoa-19-2024.txt`

- **Program**: Cử nhân ngành Khoa học Máy tính
- **Cohort**: Khóa 19 (Starting 2024)
- **URL**: https://student.uit.edu.vn/content/cu-nhan-nganh-khoa-hoc-may-tinh-ap-dung-tu-khoa-19-2024
- **Crawled Date**: 2025-10-13
- **Size**: 22,598 characters
- **Content**:
  - Complete curriculum structure
  - Required courses
  - Elective courses
  - Graduation requirements
  - Program objectives
  - Course details

## File Format

Each crawled file follows this structure:

```
URL: <source_url>
Title: <page_title>
Crawled: <timestamp>
Length: <character_count> characters
====================================================================================================
<actual_content>
```

## How to Index This Data

### 1. Install Dependencies

Make sure you have the required packages:

```bash
cd services/rag_services
pip install llama-index-vector-stores-weaviate weaviate-client
```

### 2. Configure Environment

Ensure your `.env` file has Weaviate settings:

```bash
VECTOR_BACKEND=weaviate
WEAVIATE_URL=<your_weaviate_url>
WEAVIATE_API_KEY=<your_api_key>
WEAVIATE_CLASS_NAME=ChatbotUit
```

### 3. Run Indexing Script

```bash
python scripts/index_crawled_data.py
```

This script will:
- Parse all `.txt` files in this folder
- Extract metadata (subject, year, cohort, etc.)
- Split content into chunks
- Generate embeddings
- Store in Weaviate vector database

### 4. Test Search

```bash
python scripts/test_crawled_search.py
```

## Metadata Extracted

The indexing script automatically extracts:

- **Program Level**: undergraduate, distance_learning, second_degree, etc.
- **Subject**: Khoa học Máy tính, Hệ thống Thông tin, etc.
- **Year**: Academic year (e.g., 2024)
- **Cohort**: Student cohort (e.g., Khóa 19)
- **URL**: Original source URL
- **Crawled Date**: When data was collected

## Usage in RAG System

Once indexed, this data can be queried through:

1. **Direct API**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/search \
     -H "Content-Type: application/json" \
     -d '{"query": "Chương trình đào tạo KHMT 2024"}'
   ```

2. **Orchestrator Service**: The chatbot orchestrator can use this data to answer questions about curriculum.

## Adding More Data

To add new crawled programs:

1. Save crawled content in the same format
2. Name file descriptively (use URL slug)
3. Run the indexing script
4. The system will automatically detect and process new files

## Notes

- Data is from official UIT sources
- Content is in Vietnamese
- Suitable for RAG-based Q&A about curriculum
- Can be combined with other data sources (split programs, PDFs, etc.)

## Future Improvements

- [ ] Add more programs (Hệ thống Thông tin, Mạng máy tính, etc.)
- [ ] Add different years and cohorts
- [ ] Include distance learning and second degree programs
- [ ] Add course syllabi details
- [ ] Automate periodic re-crawling for updates
