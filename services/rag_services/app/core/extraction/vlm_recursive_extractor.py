"""
VLM Recursive Extractor for Legal Document Knowledge Graph Extraction.

This module extracts Knowledge Graph nodes and relations from legal document images
(scanned PDFs) using Vision Language Models (GPT-4o or Gemini 1.5 Flash).

Key Feature: Recursive Context Handling
- Legal documents often have text that spans across pages (incomplete sentences,
  lists split between pages, etc.)
- This extractor maintains context between pages to ensure proper text continuity
- Each page extraction receives the context from the previous page

Author: Legal Document Processing Team
Date: 2024
"""

import base64
import json
import logging
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class PageContext(BaseModel):
    """
    Context state passed from one page to the next.
    
    This model captures the "state" at the end of a page to ensure
    continuity when processing the next page.
    
    Attributes:
        current_chapter: The chapter currently in effect (e.g., "Chương II: Quyền và nghĩa vụ")
        current_article: The article currently in effect (e.g., "Điều 5")
        current_clause: The clause currently in effect (e.g., "Khoản 2")
        pending_text: Text that was cut off at the end of the page and needs
                      to be continued on the next page
    """
    current_chapter: Optional[str] = Field(
        default=None,
        description="Tên chương đang hiệu lực (VD: 'Chương II: Quyền và nghĩa vụ')"
    )
    current_article: Optional[str] = Field(
        default=None,
        description="Tên điều khoản đang hiệu lực (VD: 'Điều 5')"
    )
    current_clause: Optional[str] = Field(
        default=None,
        description="Khoản đang hiệu lực (VD: 'Khoản 2')"
    )
    pending_text: Optional[str] = Field(
        default=None,
        description="Đoạn văn bản bị cắt dở ở cuối trang trước, cần nối vào đầu trang sau"
    )


class NodeType(str, Enum):
    """Types of nodes that can be extracted from legal documents."""
    DOCUMENT = "Document"
    CHAPTER = "Chapter"
    ARTICLE = "Article"
    CLAUSE = "Clause"
    POINT = "Point"
    DEFINITION = "Definition"
    SUBJECT = "Subject"
    ACTION = "Action"
    CONDITION = "Condition"
    SANCTION = "Sanction"
    REFERENCE = "Reference"
    ENTITY = "Entity"


class RelationType(str, Enum):
    """Types of relations between nodes in legal documents."""
    CONTAINS = "CONTAINS"
    DEFINES = "DEFINES"
    REFERENCES = "REFERENCES"
    REQUIRES = "REQUIRES"
    PROHIBITS = "PROHIBITS"
    PERMITS = "PERMITS"
    APPLIES_TO = "APPLIES_TO"
    SUPERSEDES = "SUPERSEDES"
    AMENDS = "AMENDS"
    FOLLOWS = "FOLLOWS"
    BELONGS_TO = "BELONGS_TO"


class ExtractedNode(BaseModel):
    """
    A node extracted from the legal document.
    
    Represents an entity or concept found in the document that will
    become a node in the Knowledge Graph.
    
    Attributes:
        id: Unique identifier for the node (e.g., "article_5", "chapter_2")
        type: The type/category of the node
        content: The actual text content of the node
        metadata: Additional metadata (page number, position, etc.)
    """
    id: str = Field(
        description="ID duy nhất cho node (VD: 'dieu_5', 'chuong_2', 'khoan_3_dieu_5')"
    )
    type: str = Field(
        description="Loại node (VD: 'Chapter', 'Article', 'Clause', 'Definition', 'Subject')"
    )
    content: str = Field(
        description="Nội dung văn bản của node"
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Metadata bổ sung (số trang, vị trí, v.v.)"
    )


class ExtractedRelation(BaseModel):
    """
    A relation between two nodes in the Knowledge Graph.
    
    Represents a directed edge from source node to target node.
    
    Attributes:
        source: ID of the source node
        target: ID of the target node
        type: The type of relation
        description: Optional description of the relation
    """
    source: str = Field(
        description="ID của node nguồn"
    )
    target: str = Field(
        description="ID của node đích"
    )
    type: str = Field(
        description="Loại quan hệ (VD: 'CONTAINS', 'REFERENCES', 'DEFINES')"
    )
    description: Optional[str] = Field(
        default=None,
        description="Mô tả chi tiết về quan hệ (nếu cần)"
    )


class PageOutput(BaseModel):
    """
    Output from processing a single page.
    
    Contains the extracted nodes and relations from the page,
    plus the context to pass to the next page.
    
    Attributes:
        nodes: List of nodes extracted from this page
        relations: List of relations extracted from this page
        next_context: Context to pass to the next page for continuity
        page_summary: Brief summary of what was found on this page
    """
    nodes: list[ExtractedNode] = Field(
        default_factory=list,
        description="Danh sách các node được trích xuất từ trang này"
    )
    relations: list[ExtractedRelation] = Field(
        default_factory=list,
        description="Danh sách các quan hệ được trích xuất từ trang này"
    )
    next_context: PageContext = Field(
        default_factory=PageContext,
        description="Context để truyền sang trang tiếp theo"
    )
    page_summary: Optional[str] = Field(
        default=None,
        description="Tóm tắt ngắn gọn nội dung trang"
    )


class ExtractionResult(BaseModel):
    """
    Final result of processing all pages in a document.
    
    Attributes:
        nodes: All nodes extracted from the entire document
        relations: All relations extracted from the entire document
        page_count: Number of pages processed
        errors: List of any errors encountered during processing
    """
    nodes: list[ExtractedNode] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
    page_count: int = Field(default=0)
    errors: list[dict] = Field(default_factory=list)


# =============================================================================
# VLM API Configuration
# =============================================================================

class VLMProvider(str, Enum):
    """Supported VLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


class VLMConfig(BaseModel):
    """Configuration for VLM API calls."""
    provider: VLMProvider = VLMProvider.OPENROUTER
    api_key: str = Field(
        default=None,
        description="API key for the VLM provider (loaded from env if not provided)"
    )
    model: str = Field(
        default=None,
        description="Model name (loaded from env if not provided)"
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for the API"
    )
    max_tokens: int = Field(default=4096, description="Maximum tokens in response")
    temperature: float = Field(default=0, description="Temperature for generation")
    max_retries: int = Field(default=3, description="Maximum retry attempts on failure")
    retry_delay: float = Field(default=2.0, description="Delay between retries in seconds")
    
    @classmethod
    def from_env(cls, provider: VLMProvider = VLMProvider.OPENROUTER) -> "VLMConfig":
        """
        Load configuration from environment variables.
        
        Environment variables:
            - OPENROUTER_API_KEY: API key for OpenRouter
            - VLM_MODEL: Model name (e.g., 'google/gemini-flash-1.5', 'openai/gpt-4o')
            - OPENAI_API_KEY: API key for OpenAI (if using OpenAI provider)
            - GEMINI_API_KEY: API key for Gemini (if using Gemini provider)
        """
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        if provider == VLMProvider.OPENROUTER:
            api_key = os.getenv("OPENROUTER_API_KEY")
            model = os.getenv("VLM_MODEL", "google/gemini-flash-1.5")
            base_url = "https://openrouter.ai/api/v1"
        elif provider == VLMProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
            model = os.getenv("VLM_MODEL", "gpt-4o")
            base_url = "https://api.openai.com/v1"
        elif provider == VLMProvider.GEMINI:
            api_key = os.getenv("GEMINI_API_KEY")
            model = os.getenv("VLM_MODEL", "gemini-1.5-flash")
            base_url = None
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        if not api_key:
            raise ValueError(
                f"API key not found. Please set the appropriate environment variable "
                f"for {provider.value} in your .env file."
            )
        
        return cls(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url
        )


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """Bạn là một chuyên gia phân tích văn bản quy phạm pháp luật Việt Nam. 
Nhiệm vụ của bạn là trích xuất Knowledge Graph từ ảnh văn bản luật.

## CONTEXT TỪ TRANG TRƯỚC
Bạn sẽ nhận được `prev_context` JSON chứa trạng thái từ trang trước:
- `current_chapter`: Chương đang hiệu lực
- `current_article`: Điều đang hiệu lực  
- `current_clause`: Khoản đang hiệu lực
- `pending_text`: Đoạn văn bản bị cắt dở ở cuối trang trước

## QUY TẮC BẮT BUỘC - RẤT QUAN TRỌNG

### QUY TẮC 1: XỬ LÝ CHUYỂN TRANG (PAGINATION CONTINUITY)
**Đây là quy tắc quan trọng nhất để tránh lỗi "Content Bleeding"**

1. **Khi đầu trang là đoạn văn bản không có tiêu đề mới:**
   - Nếu đầu trang là một đoạn văn bản KHÔNG BẮT ĐẦU bằng "Điều X", "Khoản Y", "Chương Z"
   - Đó là PHẦN TIẾP NỐI của node cuối cùng từ trang trước (trong prev_context)
   - PHẢI gán nội dung đó vào node của prev_context (current_article/current_clause)
   - KHÔNG ĐƯỢC gán vào Điều/Khoản mới tìm thấy ở GIỮA hoặc CUỐI trang

2. **Ví dụ sai (TRÁNH):**
   ```
   Trang 7 kết thúc: "...Học phí = min(HPHKC..." (thuộc Điều 4)
   Trang 8 bắt đầu: "...HK sau) + HP HK thiếu" (tiếp nối)
   Trang 8 giữa trang: "Điều 7. Chương trình đào tạo"
   
   SAI: Gán "HK sau) + HP HK thiếu" vào Điều 7
   ĐÚNG: Gán "HK sau) + HP HK thiếu" vào Điều 4 (từ prev_context)
   ```

3. **Cách xử lý đúng:**
   - Quét từ ĐẦU TRANG xuống
   - Mọi nội dung TRƯỚC KHI gặp tiêu đề mới (Điều/Khoản/Chương) → thuộc prev_context
   - Chỉ khi gặp tiêu đề mới → bắt đầu node mới

### QUY TẮC 2: LOẠI BỎ TOC/HEADER/FOOTER
**BỎ QUA hoàn toàn các nội dung sau - KHÔNG tạo node:**

1. **Mục lục (TOC):**
   - "MỤC LỤC", các dòng chỉ có tiêu đề + số trang
   - VD: "Chương I. Những quy định chung....................1"

2. **Header/Footer lặp lại:**
   - Số trang: "Trang 1/27", "1", "Page 1 of 27"
   - Tiêu đề trang lặp: Tên văn bản lặp lại ở đầu mỗi trang
   - Watermark, chữ ký điện tử

3. **Tiêu đề Chương/Điều chỉ xuất hiện như header trang:**
   - Nếu "CHƯƠNG II" xuất hiện ở đầu trang như header (không có nội dung mới theo sau)
   - → Đó là header, KHÔNG tạo node mới
   - → Chỉ tạo node khi có NỘI DUNG thực sự của Chương/Điều

### QUY TẮC 3: TRÁNH NODE TRÙNG LẶP
1. **Kiểm tra trước khi tạo node:**
   - Nếu thấy "Chương II" và prev_context.current_chapter đã là "Chương II"
   - → KHÔNG tạo node mới, đây chỉ là header trang

2. **ID phải nhất quán:**
   - Chương 2 luôn là "chuong_2" (không phải "chuong_2_to_chuc_dao_tao")
   - Điều 5 Khoản 3 luôn là "khoan_3_dieu_5"

### QUY TẮC 4: PHÂN BIỆT NỘI DUNG THỰC vs HEADER
1. **Nội dung thực của Điều/Khoản:**
   - Có văn bản chi tiết theo sau tiêu đề
   - Có các gạch đầu dòng, điểm a), b), c)
   - Có bảng, công thức, định nghĩa

2. **Header trang (không phải nội dung):**
   - Chỉ có tiêu đề, không có nội dung
   - Xuất hiện đơn lẻ ở đầu/cuối trang
   - Giống như lặp lại từ TOC

## HƯỚNG DẪN TRÍCH XUẤT

### 1. Xử lý pending_text
Nếu `pending_text` không rỗng, hãy:
- Tìm phần văn bản tiếp theo ở đầu trang hiện tại
- Nối `pending_text` với phần đầu trang để tạo thành câu/đoạn hoàn chỉnh
- Gán vào node của prev_context (current_article/current_clause)

### 2. Quét nội dung trang theo thứ tự
Với mỗi trang, quét từ TRÊN xuống DƯỚI:
1. Phần đầu trang (trước tiêu đề mới) → thuộc prev_context
2. Khi gặp tiêu đề mới (Điều X, Khoản Y) → tạo node mới
3. Nội dung sau tiêu đề → thuộc node mới đó

### 3. Trích xuất Nodes
Với mỗi thành phần pháp lý, tạo node với:
- `id`: ID duy nhất, format snake_case (VD: "dieu_5", "khoan_2_dieu_5")
- `type`: Một trong các loại:
  - "Chapter": Chương
  - "Article": Điều
  - "Clause": Khoản
  - "Point": Điểm
  - "Definition": Định nghĩa thuật ngữ
  - "Subject": Chủ thể pháp luật
  - "Table": Bảng dữ liệu
- `content`: Nội dung đầy đủ của ĐÚNG phần đó

### 4. Trích xuất Relations
Xác định quan hệ giữa các nodes:
- "CONTAINS": Chương chứa Điều, Điều chứa Khoản
- "BELONGS_TO": Khoản thuộc Điều, Điều thuộc Chương
- "REFERENCES": Tham chiếu đến điều/văn bản khác
- "DEFINES": Định nghĩa thuật ngữ

### 5. Cập nhật Context cho trang sau
Trong `next_context`, cập nhật:
- `current_chapter`: Chương cuối cùng có NỘI DUNG trên trang
- `current_article`: Điều cuối cùng có NỘI DUNG trên trang
- `current_clause`: Khoản cuối cùng có NỘI DUNG trên trang
- `pending_text`: Nếu trang kết thúc giữa chừng (câu chưa hết), lưu phần đó

## OUTPUT FORMAT
```json
{
  "nodes": [
    {"id": "string", "type": "string", "content": "string", "metadata": {"page": 1}},
    ...
  ],
  "relations": [
    {"source": "string", "target": "string", "type": "string"},
    ...
  ],
  "next_context": {
    "current_chapter": "string or null",
    "current_article": "string or null", 
    "current_clause": "string or null",
    "pending_text": "string or null"
  },
  "page_summary": "Tóm tắt ngắn nội dung trang"
}
```

## CHECKLIST TRƯỚC KHI OUTPUT
□ Nội dung đầu trang (trước tiêu đề mới) đã gán vào prev_context chưa?
□ Có bỏ qua TOC/Header/Footer chưa?
□ Có tạo node trùng với prev_context không?
□ ID có nhất quán không (chuong_2, không phải chuong_2_xyz)?
□ Content của mỗi node có ĐÚNG nội dung của nó không?

## LƯU Ý QUAN TRỌNG
1. CHỈ trả về JSON, không có text giải thích
2. Đảm bảo JSON hợp lệ (valid JSON)
3. **NỘI DUNG PHẢI ĐÚNG NODE** - Không được để nội dung của Điều này trong Điều khác
4. Luôn cập nhật next_context dù trang có nội dung hay không
"""


# =============================================================================
# Utility Functions
# =============================================================================

def encode_image_to_base64(image_path: str | Path) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
        
    Raises:
        FileNotFoundError: If the image file doesn't exist
        IOError: If there's an error reading the file
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_media_type(image_path: str | Path) -> str:
    """
    Determine the media type of an image based on its extension.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Media type string (e.g., "image/png", "image/jpeg")
    """
    path = Path(image_path)
    extension = path.suffix.lower()
    
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    
    return media_types.get(extension, "image/png")


def parse_vlm_response(response_text: str) -> PageOutput:
    """
    Parse the VLM response text into a PageOutput object.
    
    Handles various JSON formatting issues that may occur in VLM responses.
    
    Args:
        response_text: Raw text response from the VLM
        
    Returns:
        Parsed PageOutput object
        
    Raises:
        ValueError: If the response cannot be parsed as valid JSON
    """
    # Clean up the response text
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    
    try:
        data = json.loads(text)
        return PageOutput(**data)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Raw response: {text[:500]}...")
        raise ValueError(f"Invalid JSON response from VLM: {e}")


# =============================================================================
# PDF to Image Conversion
# =============================================================================

def convert_pdf_to_images(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    dpi: int = 200,
    image_format: str = "png"
) -> list[Path]:
    """
    Convert a PDF file to a list of images (one per page).
    
    This function uses pdf2image library which requires poppler to be installed.
    
    Installation:
        pip install pdf2image
        
        Windows: Download poppler from https://github.com/osber/poppler/releases
                 and add bin/ folder to PATH
        
        Linux: sudo apt-get install poppler-utils
        
        macOS: brew install poppler
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save images. If None, creates a temp directory
                    next to the PDF file
        dpi: Resolution for rendering (default: 200, higher = better quality but larger files)
        image_format: Output format - 'png' (recommended) or 'jpeg'
        
    Returns:
        List of paths to the generated image files, sorted by page number
        
    Example:
        >>> images = convert_pdf_to_images("document.pdf", dpi=300)
        >>> print(f"Generated {len(images)} page images")
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image package is required for PDF conversion. "
            "Install with: pip install pdf2image\n"
            "Also install poppler:\n"
            "  Windows: Download from https://github.com/osber/poppler/releases\n"
            "  Linux: sudo apt-get install poppler-utils\n"
            "  macOS: brew install poppler"
        )
    
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Create output directory
    if output_dir is None:
        output_dir = pdf_path.parent / f"{pdf_path.stem}_images"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting PDF to images: {pdf_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"DPI: {dpi}, Format: {image_format}")
    
    # Convert PDF to images
    try:
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            fmt=image_format,
            thread_count=4  # Use multiple threads for faster conversion
        )
    except Exception as e:
        logger.error(f"Failed to convert PDF: {e}")
        raise RuntimeError(
            f"PDF conversion failed: {e}\n"
            "Make sure poppler is installed and in PATH."
        )
    
    # Save images with zero-padded page numbers
    image_paths = []
    num_digits = len(str(len(images)))  # For zero-padding
    
    for i, image in enumerate(images, start=1):
        # Format: page_001.png, page_002.png, etc.
        filename = f"page_{str(i).zfill(num_digits)}.{image_format}"
        image_path = output_dir / filename
        
        image.save(image_path, image_format.upper())
        image_paths.append(image_path)
        
        logger.debug(f"Saved page {i}: {image_path}")
    
    logger.info(f"Successfully converted {len(images)} pages to images")
    
    return image_paths


def process_pdf_document(
    pdf_path: str | Path,
    config: VLMConfig,
    output_dir: str | Path | None = None,
    dpi: int = 200,
    keep_images: bool = True,
    continue_on_error: bool = True
) -> ExtractionResult:
    """
    Process a PDF document end-to-end: convert to images and extract Knowledge Graph.
    
    This is a convenience function that combines PDF conversion and KG extraction.
    
    Args:
        pdf_path: Path to the PDF file
        config: VLM configuration
        output_dir: Directory to save intermediate images. If None, uses temp directory
        dpi: Resolution for PDF rendering (default: 200)
        keep_images: If True, keep the generated images after processing.
                     If False, delete them to save space.
        continue_on_error: If True, continue processing even if a page fails
        
    Returns:
        ExtractionResult containing all nodes and relations
        
    Example:
        >>> config = VLMConfig.from_env()
        >>> result = process_pdf_document("legal_doc.pdf", config)
        >>> print(f"Extracted {len(result.nodes)} nodes")
    """
    import shutil
    
    pdf_path = Path(pdf_path)
    
    # Step 1: Convert PDF to images
    logger.info(f"Step 1/2: Converting PDF to images...")
    image_paths = convert_pdf_to_images(
        pdf_path=pdf_path,
        output_dir=output_dir,
        dpi=dpi
    )
    
    # Step 2: Process images
    logger.info(f"Step 2/2: Extracting Knowledge Graph from {len(image_paths)} pages...")
    result = process_document_images(
        image_paths=image_paths,
        config=config,
        continue_on_error=continue_on_error
    )
    
    # Clean up images if requested
    if not keep_images and output_dir:
        logger.info("Cleaning up temporary images...")
        shutil.rmtree(output_dir, ignore_errors=True)
    
    return result


# =============================================================================
# VLM API Callers
# =============================================================================

def call_openai_api(
    image_base64: str,
    media_type: str,
    prev_context: PageContext,
    config: VLMConfig,
    page_number: int
) -> str:
    """
    Call OpenAI's GPT-4o Vision API.
    
    Args:
        image_base64: Base64 encoded image
        media_type: Image media type
        prev_context: Context from previous page
        config: VLM configuration
        page_number: Current page number (for logging)
        
    Returns:
        Raw text response from the API
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is required. Install with: pip install openai")
    
    client = OpenAI(api_key=config.api_key)
    
    # Build the user message with context
    user_content = [
        {
            "type": "text",
            "text": f"""Đây là trang {page_number} của văn bản pháp luật.

## CONTEXT TỪ TRANG TRƯỚC:
```json
{prev_context.model_dump_json(indent=2)}
```

Hãy phân tích ảnh và trích xuất Knowledge Graph theo hướng dẫn trong system prompt.
Nhớ xử lý pending_text nếu có và cập nhật next_context cho trang sau."""
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{image_base64}",
                "detail": "high"
            }
        }
    ]
    
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        max_tokens=config.max_tokens,
        temperature=config.temperature
    )
    
    return response.choices[0].message.content


def call_openrouter_api(
    image_base64: str,
    media_type: str,
    prev_context: PageContext,
    config: VLMConfig,
    page_number: int
) -> str:
    """
    Call OpenRouter API with vision-capable models.
    
    OpenRouter provides a unified API to access multiple VLM providers
    including GPT-4o, Gemini, Claude, etc.
    
    Args:
        image_base64: Base64 encoded image
        media_type: Image media type
        prev_context: Context from previous page
        config: VLM configuration
        page_number: Current page number (for logging)
        
    Returns:
        Raw text response from the API
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is required. Install with: pip install openai")
    
    # OpenRouter uses OpenAI-compatible API
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url or "https://openrouter.ai/api/v1"
    )
    
    # Build the user message with context
    user_content = [
        {
            "type": "text",
            "text": f"""Đây là trang {page_number} của văn bản pháp luật.

## CONTEXT TỪ TRANG TRƯỚC:
```json
{prev_context.model_dump_json(indent=2)}
```

Hãy phân tích ảnh và trích xuất Knowledge Graph theo hướng dẫn trong system prompt.
Nhớ xử lý pending_text nếu có và cập nhật next_context cho trang sau."""
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{image_base64}"
            }
        }
    ]
    
    response = client.chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        extra_headers={
            "HTTP-Referer": "https://github.com/LiamNNT/Chatbot-UIT",
            "X-Title": "Legal Document KG Extractor"
        }
    )
    
    return response.choices[0].message.content


def call_gemini_api(
    image_base64: str,
    media_type: str,
    prev_context: PageContext,
    config: VLMConfig,
    page_number: int
) -> str:
    """
    Call Google's Gemini Vision API.
    
    Args:
        image_base64: Base64 encoded image
        media_type: Image media type
        prev_context: Context from previous page
        config: VLM configuration
        page_number: Current page number (for logging)
        
    Returns:
        Raw text response from the API
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai package is required. "
            "Install with: pip install google-generativeai"
        )
    
    genai.configure(api_key=config.api_key)
    
    model = genai.GenerativeModel(
        model_name=config.model,
        system_instruction=SYSTEM_PROMPT
    )
    
    # Decode base64 to bytes for Gemini
    image_bytes = base64.b64decode(image_base64)
    
    # Build the prompt
    prompt = f"""Đây là trang {page_number} của văn bản pháp luật.

## CONTEXT TỪ TRANG TRƯỚC:
```json
{prev_context.model_dump_json(indent=2)}
```

Hãy phân tích ảnh và trích xuất Knowledge Graph theo hướng dẫn trong system prompt.
Nhớ xử lý pending_text nếu có và cập nhật next_context cho trang sau."""
    
    # Create the image part
    image_part = {
        "mime_type": media_type,
        "data": image_bytes
    }
    
    response = model.generate_content(
        [prompt, image_part],
        generation_config=genai.GenerationConfig(
            max_output_tokens=config.max_tokens,
            temperature=config.temperature
        )
    )
    
    return response.text


def call_vlm_api(
    image_base64: str,
    media_type: str,
    prev_context: PageContext,
    config: VLMConfig,
    page_number: int
) -> str:
    """
    Call the appropriate VLM API based on configuration.
    
    This is the main API dispatcher that routes to the correct provider.
    
    Args:
        image_base64: Base64 encoded image
        media_type: Image media type
        prev_context: Context from previous page
        config: VLM configuration
        page_number: Current page number
        
    Returns:
        Raw text response from the API
    """
    if config.provider == VLMProvider.OPENROUTER:
        return call_openrouter_api(
            image_base64, media_type, prev_context, config, page_number
        )
    elif config.provider == VLMProvider.OPENAI:
        return call_openai_api(
            image_base64, media_type, prev_context, config, page_number
        )
    elif config.provider == VLMProvider.GEMINI:
        return call_gemini_api(
            image_base64, media_type, prev_context, config, page_number
        )
    else:
        raise ValueError(f"Unsupported VLM provider: {config.provider}")


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_single_page(
    image_path: str | Path,
    prev_context: PageContext,
    config: VLMConfig,
    page_number: int
) -> PageOutput:
    """
    Process a single page image and extract Knowledge Graph data.
    
    Args:
        image_path: Path to the page image
        prev_context: Context from the previous page
        config: VLM configuration
        page_number: Page number (1-indexed)
        
    Returns:
        PageOutput containing extracted nodes, relations, and next context
        
    Raises:
        Exception: If processing fails after all retries
    """
    logger.info(f"Processing page {page_number}: {image_path}")
    
    # Encode image
    image_base64 = encode_image_to_base64(image_path)
    media_type = get_image_media_type(image_path)
    
    last_error = None
    
    for attempt in range(1, config.max_retries + 1):
        try:
            # Call VLM API
            response_text = call_vlm_api(
                image_base64=image_base64,
                media_type=media_type,
                prev_context=prev_context,
                config=config,
                page_number=page_number
            )
            
            # Parse response
            page_output = parse_vlm_response(response_text)
            
            # Add page number to node metadata
            for node in page_output.nodes:
                if node.metadata is None:
                    node.metadata = {}
                node.metadata["page_number"] = page_number
            
            logger.info(
                f"Page {page_number}: Extracted {len(page_output.nodes)} nodes, "
                f"{len(page_output.relations)} relations"
            )
            
            return page_output
            
        except Exception as e:
            last_error = e
            logger.warning(
                f"Page {page_number}, attempt {attempt}/{config.max_retries} failed: {e}"
            )
            
            if attempt < config.max_retries:
                logger.info(f"Retrying in {config.retry_delay} seconds...")
                time.sleep(config.retry_delay)
    
    # All retries failed
    raise Exception(
        f"Failed to process page {page_number} after {config.max_retries} attempts. "
        f"Last error: {last_error}"
    )


def process_document_images(
    image_paths: list[str | Path],
    config: VLMConfig,
    continue_on_error: bool = True
) -> ExtractionResult:
    """
    Process all pages of a document and extract complete Knowledge Graph.
    
    This is the main entry point for processing a legal document.
    It processes pages sequentially, maintaining context between pages
    to handle text that spans across page boundaries.
    
    Args:
        image_paths: List of paths to page images, sorted by page order
        config: VLM configuration
        continue_on_error: If True, continue processing even if a page fails.
                          If False, raise exception on first failure.
        
    Returns:
        ExtractionResult containing all nodes and relations from the document
        
    Example:
        >>> config = VLMConfig(
        ...     provider=VLMProvider.OPENAI,
        ...     api_key="your-api-key",
        ...     model="gpt-4o"
        ... )
        >>> image_paths = ["page_001.png", "page_002.png", "page_003.png"]
        >>> result = process_document_images(image_paths, config)
        >>> print(f"Extracted {len(result.nodes)} nodes")
    """
    logger.info(f"Starting document processing with {len(image_paths)} pages")
    
    # Initialize result containers
    all_nodes: list[ExtractedNode] = []
    all_relations: list[ExtractedRelation] = []
    errors: list[dict] = []
    
    # Initialize empty context for first page
    current_context = PageContext()
    
    # Process each page sequentially
    for page_num, image_path in enumerate(image_paths, start=1):
        try:
            # Process the page
            page_output = process_single_page(
                image_path=image_path,
                prev_context=current_context,
                config=config,
                page_number=page_num
            )
            
            # Collect nodes and relations
            all_nodes.extend(page_output.nodes)
            all_relations.extend(page_output.relations)
            
            # Update context for next page
            current_context = page_output.next_context
            
            logger.debug(
                f"Page {page_num} context: chapter={current_context.current_chapter}, "
                f"article={current_context.current_article}, "
                f"pending={bool(current_context.pending_text)}"
            )
            
        except Exception as e:
            error_info = {
                "page_number": page_num,
                "image_path": str(image_path),
                "error": str(e)
            }
            errors.append(error_info)
            logger.error(f"Error processing page {page_num}: {e}")
            
            if not continue_on_error:
                raise
            
            # Continue with empty context update on error
            logger.warning(f"Continuing with previous context after error on page {page_num}")
    
    # Build final result
    result = ExtractionResult(
        nodes=all_nodes,
        relations=all_relations,
        page_count=len(image_paths),
        errors=errors
    )
    
    logger.info(
        f"Document processing complete. "
        f"Total: {len(all_nodes)} nodes, {len(all_relations)} relations, "
        f"{len(errors)} errors"
    )
    
    return result


def deduplicate_nodes(nodes: list[ExtractedNode]) -> list[ExtractedNode]:
    """
    Remove duplicate nodes based on their IDs.
    
    When the same entity spans multiple pages, it may be extracted
    multiple times. This function keeps only the first occurrence
    or merges content if IDs match.
    
    Args:
        nodes: List of extracted nodes (may contain duplicates)
        
    Returns:
        Deduplicated list of nodes
    """
    seen_ids: dict[str, ExtractedNode] = {}
    
    for node in nodes:
        if node.id not in seen_ids:
            seen_ids[node.id] = node
        else:
            # Merge: keep the one with longer content
            existing = seen_ids[node.id]
            if len(node.content) > len(existing.content):
                seen_ids[node.id] = node
    
    return list(seen_ids.values())


def merge_extraction_results(results: list[ExtractionResult]) -> ExtractionResult:
    """
    Merge multiple extraction results into one.
    
    Useful when processing a document in batches or combining
    results from multiple documents.
    
    Args:
        results: List of ExtractionResult objects
        
    Returns:
        Merged ExtractionResult
    """
    all_nodes = []
    all_relations = []
    total_pages = 0
    all_errors = []
    
    for result in results:
        all_nodes.extend(result.nodes)
        all_relations.extend(result.relations)
        total_pages += result.page_count
        all_errors.extend(result.errors)
    
    # Deduplicate nodes
    unique_nodes = deduplicate_nodes(all_nodes)
    
    # Deduplicate relations
    seen_relations = set()
    unique_relations = []
    for rel in all_relations:
        key = (rel.source, rel.target, rel.type)
        if key not in seen_relations:
            seen_relations.add(key)
            unique_relations.append(rel)
    
    return ExtractionResult(
        nodes=unique_nodes,
        relations=unique_relations,
        page_count=total_pages,
        errors=all_errors
    )


def save_result_to_json(result: ExtractionResult, output_path: str | Path) -> None:
    """
    Save extraction result to a JSON file.
    
    Args:
        result: Extraction result to save
        output_path: Path for the output JSON file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    
    logger.info(f"Result saved to {output_path}")


def load_result_from_json(input_path: str | Path) -> ExtractionResult:
    """
    Load extraction result from a JSON file.
    
    Args:
        input_path: Path to the input JSON file
        
    Returns:
        Loaded ExtractionResult
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return ExtractionResult(**data)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """
    Command-line interface for the VLM Recursive Extractor.
    
    Usage:
        # Process PDF directly (recommended)
        python vlm_recursive_extractor.py --pdf document.pdf --output result.json
        
        # Process existing images
        python vlm_recursive_extractor.py --input-dir ./images --output result.json
        
    Environment variables (in .env file):
        OPENROUTER_API_KEY: API key for OpenRouter
        VLM_MODEL: Model name (e.g., 'google/gemini-flash-1.5', 'openai/gpt-4o')
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract Knowledge Graph from legal documents (PDF or images) using VLM"
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pdf",
        type=str,
        help="Path to PDF file to process"
    )
    input_group.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing page images (if already converted)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="extraction_result.json",
        help="Output JSON file path (default: extraction_result.json)"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory to save converted images (default: <pdf_name>_images/)"
    )
    
    # PDF conversion options
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for PDF to image conversion (default: 200, higher = better quality)"
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        default=True,
        help="Keep converted images after processing (default: True)"
    )
    parser.add_argument(
        "--delete-images",
        action="store_true",
        help="Delete converted images after processing to save space"
    )
    
    # VLM options
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openrouter", "openai", "gemini"],
        default="openrouter",
        help="VLM provider (default: openrouter)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per page (default: 3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config from environment
    provider = VLMProvider(args.provider)
    
    try:
        config = VLMConfig.from_env(provider=provider)
        config.max_retries = args.max_retries
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease create a .env file with:")
        print("  OPENROUTER_API_KEY=your-api-key")
        print("  VLM_MODEL=google/gemini-flash-1.5")
        return 1
    
    print(f"Using provider: {config.provider.value}")
    print(f"Using model: {config.model}")
    print()
    
    # Process based on input type
    if args.pdf:
        # Process PDF directly
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"Error: PDF file not found: {pdf_path}")
            return 1
        
        print(f"Processing PDF: {pdf_path}")
        
        keep_images = not args.delete_images
        
        try:
            result = process_pdf_document(
                pdf_path=pdf_path,
                config=config,
                output_dir=args.image_dir,
                dpi=args.dpi,
                keep_images=keep_images
            )
        except ImportError as e:
            print(f"Error: {e}")
            return 1
        except RuntimeError as e:
            print(f"Error: {e}")
            return 1
    else:
        # Process existing images
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            return 1
        
        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
        image_paths = sorted([
            p for p in input_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ])
        
        if not image_paths:
            print(f"Error: No images found in {input_dir}")
            return 1
        
        print(f"Found {len(image_paths)} images to process")
        
        result = process_document_images(image_paths, config)
    
    # Save result
    save_result_to_json(result, args.output)
    
    # Print summary
    print(f"\n{'='*50}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*50}")
    print(f"Pages processed: {result.page_count}")
    print(f"Nodes extracted: {len(result.nodes)}")
    print(f"Relations extracted: {len(result.relations)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Output saved to: {args.output}")
    
    if result.errors:
        print("\nErrors encountered:")
        for error in result.errors:
            print(f"  - Page {error['page_number']}: {error['error']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
