"""
Quick test script to debug the hybrid extractor VLM call.
"""
import os
import sys
import json
import base64
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_structure_extraction():
    """Test the full structure extraction flow."""
    from openai import OpenAI
    
    # Config
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("VLM_MODEL", "google/gemini-flash-1.5")
    
    print(f"API Key exists: {bool(api_key)}")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    
    # Load image
    image_dir = Path(__file__).parent.parent / "data" / "quy_dinh" / "1393-qd-dhcntt_29-12-2023_cap_nhat_quy_che_dao_tao_theo_hoc_che_tin_chi_cho_he_dai_hoc_chinh_quy_images"
    
    if not image_dir.exists():
        print(f"ERROR: Image dir not found: {image_dir}")
        return
    
    image_path = list(image_dir.glob("*.png"))[0]
    print(f"Using image: {image_path}")
    
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    # Create prompt - similar to hybrid_extractor
    prompt = """Bạn là chuyên gia phân tích cấu trúc văn bản pháp luật.

Từ ảnh trang văn bản pháp luật, hãy OCR và trích xuất cấu trúc.

YÊU CẦU: CHỈ trả về JSON object, KHÔNG có markdown, KHÔNG có giải thích.

Format JSON:
{"nodes": [{"id": "dieu_1", "type": "Article", "title": "Điều 1...", "full_text": "nội dung...", "page_number": 1}], "relations": [], "next_context": {"current_article": null, "pending_text": null}}"""

    # Call API
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    print("\nCalling VLM API...")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }],
            max_tokens=4096,
            temperature=0,
            extra_headers={
                "HTTP-Referer": "https://github.com/LiamNNT/Chatbot-UIT",
                "X-Title": "Legal Document KG Extractor"
            }
        )
        
        result = response.choices[0].message.content
        print(f"\n=== RAW RESPONSE ({len(result)} chars) ===")
        print(result)
        print("=" * 50)
        
        # Try to parse JSON
        print("\n=== PARSING JSON ===")
        
        text = result.strip()
        
        # Remove markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        
        # Find JSON object
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]
        
        print(f"Text to parse:\n{text[:500]}...")
        
        try:
            data = json.loads(text)
            print(f"\nParsed successfully!")
            print(f"Nodes: {len(data.get('nodes', []))}")
            print(f"Relations: {len(data.get('relations', []))}")
            
            for node in data.get('nodes', []):
                print(f"  - {node.get('id')}: {node.get('type')} - {node.get('title', '')[:50]}")
                
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"At position {e.pos}: {text[max(0, e.pos-20):e.pos+20]}")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_structure_extraction()
