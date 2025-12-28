#!/usr/bin/env python3
"""
Server startup script for Vietnamese Hybrid RAG system
"""
import sys
import os
import uvicorn
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Set environment variables
os.environ.setdefault("PYTHONPATH", str(current_dir))

def main():
    """Start the server"""
    print("🚀 Starting Vietnamese Hybrid RAG Server...")
    print("🏗️  Architecture: Ports & Adapters (Clean Architecture)")
    print("🔍 OpenSearch integration enabled")
    print(f"📁 Working directory: {current_dir}")
    
    try:
        # Import and start the app
        from app.main import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Server startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
