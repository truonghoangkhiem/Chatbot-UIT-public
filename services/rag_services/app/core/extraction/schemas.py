"""
Schema definitions for Hybrid Two-Stage Knowledge Graph Extraction Pipeline.

This module contains all Pydantic models, enums, and configuration classes
used by the hybrid extractor for legal document processing.

Author: Legal Document Processing Team
Date: 2024
"""

from enum import Enum
from typing import List, Literal, Optional, Dict, Any

from pydantic import BaseModel, Field


# =============================================================================
# Structural Extraction Models (Stage 1)
# =============================================================================

class StructureNodeType(str, Enum):
    """Types of structural nodes in legal documents."""
    DOCUMENT = "Document"
    CHAPTER = "Chapter"
    ARTICLE = "Article"
    CLAUSE = "Clause"
    POINT = "Point"
    TABLE = "Table"  # [NEW] Node loại Bảng


class StructureNode(BaseModel):
    """
    Structural node extracted from document pages (Stage 1).
    """
    id: str = Field(description="Unique ID (e.g., 'dieu_5', 'chuong_2', 'table_1_dieu_5')")
    type: StructureNodeType = Field(description="Type of structure node")
    title: str = Field(description="Title or heading (e.g., 'Điều 5', 'Bảng quy đổi')")
    full_text: str = Field(description="Complete text content. For Tables, this MUST be Markdown.")
    page_range: List[int] = Field(default_factory=list, description="Pages where this node appears")
    parent_id: Optional[str] = Field(default=None, description="ID of parent node")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StructureRelation(BaseModel):
    """Relation between structural nodes."""
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    type: str = Field(default="CONTAINS", description="Relation type (CONTAINS, FOLLOWS)")


class StructureExtractionResult(BaseModel):
    """Result of Stage 1: Structural Extraction."""
    document: Optional[StructureNode] = Field(default=None)
    chapters: List[StructureNode] = Field(default_factory=list)
    articles: List[StructureNode] = Field(default_factory=list)
    clauses: List[StructureNode] = Field(default_factory=list)
    tables: List[StructureNode] = Field(default_factory=list)  # [NEW] Danh sách bảng riêng biệt
    relations: List[StructureRelation] = Field(default_factory=list)
    page_count: int = Field(default=0)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# Semantic Extraction Models (Stage 2)
# =============================================================================

class SemanticNode(BaseModel):
    """Semantic entity extracted from text."""
    id: str = Field(description="Unique ID")
    type: str = Field(description="Entity type matching NodeCategory")
    text: str = Field(description="Entity text as appears in document")
    normalized: Optional[str] = Field(default=None, description="Normalized form")
    confidence: float = Field(default=0.9)
    source_article_id: str = Field(description="ID of the article this was extracted from")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SemanticRelation(BaseModel):
    """Semantic relation between entities."""
    source_id: str = Field(description="Source entity ID")
    target_id: str = Field(description="Target entity ID")
    type: str = Field(description="Relation type matching RelationshipType")
    confidence: float = Field(default=0.9)
    evidence: str = Field(default="", description="Text evidence for this relation")
    source_article_id: str = Field(description="ID of the article this was extracted from")


class Modification(BaseModel):
    """
    Model to capture legal changes/modifications between documents.
    
    Represents amendments, replacements, supplements, or repeals
    that one legal document makes to another.
    """
    action: Literal["AMENDS", "REPLACES", "SUPPLEMENTS", "REPEALS"] = Field(
        description="Type of modification action"
    )
    source_text_id: str = Field(
        description="ID of the extracting node (e.g., 'dieu_1')"
    )
    target_document_signature: str = Field(
        description="Signature of the target document (e.g., '790/QĐ-ĐHCNTT')"
    )
    target_article: Optional[str] = Field(
        default=None,
        description="Target article being modified (e.g., 'Điều 4')"
    )
    target_clause: Optional[str] = Field(
        default=None,
        description="Target clause being modified (e.g., 'Khoản 3')"
    )
    effective_date: Optional[str] = Field(
        default=None,
        description="Date when the modification takes effect"
    )
    description: str = Field(
        description="Summary of the change"
    )


class SemanticExtractionResult(BaseModel):
    """Result of Stage 2 for a single article."""
    article_id: str
    nodes: List[SemanticNode] = Field(default_factory=list)
    relations: List[SemanticRelation] = Field(default_factory=list)
    modifications: List[Modification] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


# =============================================================================
# Combined Result Models
# =============================================================================

class HybridExtractionResult(BaseModel):
    """Final result combining structure and semantics."""
    structure: StructureExtractionResult
    semantic_nodes: List[SemanticNode] = Field(default_factory=list)
    semantic_relations: List[SemanticRelation] = Field(default_factory=list)
    modifications: List[Modification] = Field(default_factory=list)
    total_pages: int = Field(default=0)
    total_articles_processed: int = Field(default=0)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


class PageContext(BaseModel):
    """Context passed between pages."""
    current_chapter: Optional[str] = None
    current_chapter_id: Optional[str] = None
    current_article: Optional[str] = None
    current_article_id: Optional[str] = None
    current_clause: Optional[str] = None
    pending_text: Optional[str] = None
    pending_node_id: Optional[str] = None


# =============================================================================
# Configuration Models
# =============================================================================

class VLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


class VLMConfig(BaseModel):
    provider: VLMProvider = VLMProvider.OPENROUTER
    api_key: Optional[str] = None
    model: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 4096
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay: float = 2.0

    @classmethod
    def from_env(cls, provider: VLMProvider = VLMProvider.OPENROUTER) -> "VLMConfig":
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        if provider == VLMProvider.OPENROUTER:
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            model = os.getenv("VLM_MODEL", "openai/gpt-4o-mini")
            base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        elif provider == VLMProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("VLM_MODEL", "gpt-4o")
            base_url = "https://api.openai.com/v1"
        elif provider == VLMProvider.GEMINI:
            api_key = os.getenv("GEMINI_API_KEY")
            model = os.getenv("VLM_MODEL", "gemini-1.5-flash")
            base_url = ""
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        if not api_key:
            raise ValueError(f"API key not found for {provider.value}.")
        return cls(provider=provider, api_key=api_key, model=model, base_url=base_url)


class LLMConfig(BaseModel):
    provider: str = "openrouter"
    api_key: Optional[str] = None
    model: str = "openai/gpt-4o-mini"
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 2000
    temperature: float = 0.0
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        model = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
        base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        
        if not api_key:
            raise ValueError("API key not found for LLM.")
        return cls(api_key=api_key, model=model, base_url=base_url)


# =============================================================================
# Definitions & Prompts
# =============================================================================

VALID_ENTITY_TYPES = {
    "MON_HOC", "QUY_DINH", "DIEU_KIEN", "SINH_VIEN", "KHOA",
    "KY_HOC", "HOC_PHI", "DIEM_SO", "TIN_CHI", "THOI_GIAN",
    "NGANH", "CHUONG_TRINH_DAO_TAO", "GIANG_VIEN",
    "CHUNG_CHI", "DO_KHO", "DOI_TUONG", "TAI_CHINH", "DIEU_KIEN_SO",
    # Extended types for amendment documents
    "VAN_BAN", "SO_LUONG", "MUC_PHI", "DIEU_KHOAN"
}

VALID_RELATION_TYPES = {
    "YEU_CAU", "DIEU_KIEN_TIEN_QUYET", "AP_DUNG_CHO",
    "QUY_DINH_DIEU_KIEN", "THUOC_KHOA", "HOC_TRONG",
    "YEU_CAU_DIEU_KIEN", "LIEN_QUAN_NOI_DUNG", "CUA_NGANH",
    "THUOC_CHUONG_TRINH", "QUAN_LY",
    "DAT_DIEM", "TUONG_DUONG", "MIEN_GIAM", "CHI_PHOI", "THUOC_VE",
    # Extended relation types
    "GIOI_HAN", "SUA_DOI", "THAY_THE", "BO_SUNG", "BAI_BO"
}

UNIFIED_ACADEMIC_SCHEMA = """
- **LOẠI THỰC THỂ (Entity Types):**
  + **MON_HOC**: Môn học, học phần (VD: "Anh văn 1", "ENG01", "môn học đại cương").
  + **CHUNG_CHI**: Chứng chỉ (VD: "TOEIC", "IELTS", "MOS").
  + **DIEM_SO**: Điểm số cụ thể (VD: "450", "5.0", "điểm M").
  + **DO_KHO**: Cấp độ/Trình độ (VD: "B1", "Bậc 3/6").
  + **DOI_TUONG**: Nhóm sinh viên áp dụng (VD: "Chương trình tiên tiến", "Hệ CLC", "sinh viên chính quy").
  + **QUY_DINH**: Tên quy định, điều khoản (VD: "Điều 5", "Quyết định 790").
  + **THOI_GIAN**: Thời hạn, thời gian (VD: "đầu khóa", "2 năm", "1 tháng", "học kỳ hè").
  + **KHOA**: Đơn vị quản lý (VD: "P.ĐTĐH", "Phòng Đào tạo").
  + **NGANH**: Ngành học.
  + **SO_LUONG**: Số lượng cụ thể (VD: "70 sinh viên", "12 tín chỉ").
  + **TIN_CHI**: Số tín chỉ (VD: "12 tín chỉ", "TCHPHM").
  + **HOC_PHI**: Học phí, mức phí (VD: "HPTCHM", "học phí học kỳ hè").
  + **DIEU_KIEN**: Điều kiện áp dụng (VD: "đăng ký tối thiểu", "đã thi đạt").
  + **VAN_BAN**: Văn bản pháp lý (VD: "Quyết định 790/QĐ-ĐHCNTT", "Quy chế đào tạo").

- **LOẠI QUAN HỆ (Relation Types):**
  + **DAT_DIEM**: (Chứng chỉ/Môn học) -> đạt mức -> (Điểm số).
  + **TUONG_DUONG**: (Thực thể A) -> tương đương -> (Thực thể B).
  + **MIEN_GIAM**: (Điều kiện) -> giúp miễn/giảm -> (Môn học).
  + **YEU_CAU**: (Quy định) -> yêu cầu bắt buộc -> (Điều kiện/Số lượng).
  + **AP_DUNG_CHO**: (Quy định) -> áp dụng cho -> (Đối tượng).
  + **QUY_DINH_DIEU_KIEN**: (Điều) -> quy định chi tiết -> (Điều kiện).
  + **GIOI_HAN**: (Quy định) -> giới hạn -> (Số lượng/Thời gian).
  + **THUOC_VE**: (Thực thể con) -> thuộc về -> (Thực thể cha).
"""

# [UPDATED] Structure Extraction Prompt - Dạy VLM nhận diện văn bản sửa đổi
# [CRITICAL UPDATE] Structure Extraction Prompt - Fix Page Boundary Logic
# [CRITICAL UPDATE] Structure Extraction Prompt - Fix Page Boundary Logic
STRUCTURE_EXTRACTION_PROMPT = """
You are an expert AI specializing in Vietnamese Legal Document Structure Extraction.
Your task is to analyze the provided document page and extract hierarchical structure into strict JSON.

## 1. CRITICAL ALGORITHM: PAGE BREAK HANDLING (MUST FOLLOW)
You must process the page text in this EXACT order to prevent content errors:

**STEP 1: Identify the FIRST Header**
   - Scan down the page to find the *first* occurrence of "Chương", "Điều", or "Khoản" (e.g., "Điều 7").
   - Define `split_point` at that header.

**STEP 2: Process Top Content (Orphan Text)**
   - **IF** there is text *above* the `split_point` (and it's not a Page Header/TOC):
     - This text **BELONGS TO THE PREVIOUS NODE** (from `prev_context`).
     - **ACTION**: Put this text into `next_context.pending_text` or append it mentally to the last active node ID from `prev_context`.
     - **DO NOT** include this top text in the `full_text` of the new node found in Step 3.
     - **DO NOT** create a new "phantom" node for this text.

**STEP 3: Process New Nodes**
   - Start creating new nodes ONLY from the `split_point` downwards.
   - Example: If Page 8 starts with text from Article 4, and then "Điều 7" appears halfway down:
     - Text top -> prev_context (Article 4)
     - "Điều 7..." -> New Node (Article 7)

## 2. EXTRACTION RULES
1. **Chapter (Chương)**: Starts with "Chương" + Roman numerals.
2. **Article (Điều)**: Starts with "Điều" + Number.
3. **Clause (Khoản)**: Number + dot (e.g., "1.", "2.") OR Amendment style "Khoản X Điều Y".
4. **Table (Bảng)**: Convert strictly to **Markdown Table**.

## 3. STRICT OUTPUT SCHEMA (JSON ONLY)
{{
  "nodes": [
    {{
      "id": "snake_case_id",
      "type": "Article" | "Clause" | "Table",
      "title": "Title (e.g., 'Điều 5', 'Khoản 1')",
      "full_text": "Content starting FROM the title downwards. DO NOT include text from top of page if it belongs to previous article.",
      "page_number": 1
    }}
  ],
  "relations": [
    {{ "source": "parent_id", "target": "child_id", "type": "CONTAINS" }}
  ],
  "next_context": {{
    "current_article_id": "ID of the last active article on this page",
    "pending_text": "Any text at the very bottom of the page that is cut off mid-sentence"
  }}
}}

## CONTEXT FROM PREVIOUS PAGE (Use this for Step 2)
Last Active Node ID: {prev_context.get('current_article_id')}
"""

# [CRITICAL UPDATE] Semantic Extraction Prompt - Anti-Hallucination Logic
SEMANTIC_EXTRACTION_PROMPT = """Bạn là chuyên gia xây dựng Knowledge Graph (KG) pháp luật.

## NHIỆM VỤ
Trích xuất Thực thể (Entities), Quan hệ (Relations), và **Sửa đổi (Modifications)**.

##QUY TẮC SỐ 1: LOGIC XÁC ĐỊNH VĂN BẢN (ANTI-HALLUCINATION)
Trước khi trích xuất `modifications`, bạn phải xác định loại văn bản hiện tại.

**TRƯỜNG HỢP 1: VĂN BẢN GỐC (Base Regulation)**
- **Dấu hiệu:** Tiêu đề là "QUY CHẾ...", "QUY ĐỊNH...", "LUẬT...". Nội dung đưa ra các định nghĩa, quy tắc mới.
- **Hành động:** `modifications` = [] (Mảng rỗng).
- **Lý do:** Văn bản gốc KHÔNG sửa đổi chính nó. Nó thiết lập quy định mới.
- *Ví dụ:* "Quy chế 790" -> modifications: []

**TRƯỜNG HỢP 2: VĂN BẢN SỬA ĐỔI (Amendment Document)**
- **Dấu hiệu:** Tiêu đề chứa "Sửa đổi, bổ sung", "Cập nhật". Nội dung ghi rõ "Sửa đổi Khoản X Điều Y của Quyết định Z".
- **Hành động:** Trích xuất vào `modifications`.
- **Điều kiện bắt buộc:** `target_document_signature` PHẢI KHÁC `source_document_id`.

## 2. HƯỚNG DẪN TRÍCH XUẤT CHI TIẾT

### A. Thực thể & Quan hệ (Luôn thực hiện)
- Trích xuất: MON_HOC, HOC_PHI, TIN_CHI, DIEM_SO, THOI_GIAN.
- Quan hệ: YEU_CAU, GIOI_HAN, AP_DUNG_CHO.

### B. Modifications (Chỉ thực hiện nếu là TRƯỜNG HỢP 2)
Nếu phát hiện câu: "Sửa đổi Khoản 3 Điều 4 của Quyết định 790/QĐ-ĐHCNTT":
- action: "AMENDS"
- target_document_signature: "790/QĐ-ĐHCNTT"
- target_article: "Điều 4"
- target_clause: "Khoản 3"

## INPUT DATA
ID: {article_id}
Title: {article_title}
Text:
{article_text}

## OUTPUT FORMAT (JSON ONLY)
{{
  "document_type": "ORIGINAL" | "AMENDMENT",  <-- Hãy xác định loại trước
  "entities": [...],
  "relations": [...],
  "modifications": [
    // TUYỆT ĐỐI ĐỂ TRỐNG NẾU document_type LÀ "ORIGINAL"
  ]
}}
"""