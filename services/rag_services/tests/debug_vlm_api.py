"""
Debug script to test VLM API call directly.
"""
import os
import base64
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

def test_vlm_api():
    """Test VLM API call directly."""
    from openai import OpenAI
    
    # Load config
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("VLM_MODEL", "google/gemini-flash-1.5")
    
    print(f"API Key: {api_key[:20]}..." if api_key else "None")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    
    if not api_key:
        print("ERROR: No API key found!")
        return
    
    # Load test image
    image_dir = Path(__file__).parent.parent / "data" / "quy_dinh" / "1393-qd-dhcntt_29-12-2023_cap_nhat_quy_che_dao_tao_theo_hoc_che_tin_chi_cho_he_dai_hoc_chinh_quy_images"
    
    if not image_dir.exists():
        print(f"Image dir not found: {image_dir}")
        # Try to find existing images
        alt_dir = Path(__file__).parent.parent / "data" / "quy_dinh"
        for d in alt_dir.iterdir():
            if d.is_dir() and "images" in d.name:
                image_dir = d
                break
    
    print(f"Looking for images in: {image_dir}")
    
    if not image_dir.exists():
        print("No image directory found. Please run the main script first to convert PDF.")
        return
    
    images = list(image_dir.glob("*.png"))
    if not images:
        print("No PNG images found")
        return
    
    image_path = images[0]
    print(f"Using image: {image_path}")
    
    # Encode image
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    print(f"Image encoded: {len(image_base64)} chars")
    
    # Create client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # Simple test prompt
    prompt = """Đây là một trang văn bản pháp luật. Hãy OCR và mô tả ngắn gọn nội dung trong 2-3 câu."""
    
    print(f"\nCalling API with model: {model}")
    print("-" * 50)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
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
                }
            ],
            max_tokens=1000,
            temperature=0,
            extra_headers={
                "HTTP-Referer": "https://github.com/LiamNNT/Chatbot-UIT",
                "X-Title": "Legal Document KG Extractor"
            }
        )
        
        result = response.choices[0].message.content
        print("Response:")
        print(result)
        print("-" * 50)
        print(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_vlm_api()
