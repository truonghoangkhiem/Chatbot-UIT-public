#!/usr/bin/env python3
"""
Stop all backend services for Chatbot-UIT

Usage:
    python stop_backend.py
"""

import subprocess
import sys
from pathlib import Path

def print_info(text: str):
    print(f"ℹ {text}")

def print_success(text: str):
    print(f"✓ {text}")

def kill_port(port: int):
    """Kill process on a specific port"""
    print_info(f"Stopping process on port {port}...")
    subprocess.run(
        ["bash", "-c", f"lsof -ti:{port} | xargs kill -9 2>/dev/null || true"],
        check=False
    )

def stop_docker_services(project_root: Path):
    """Stop Docker services"""
    print_info("Stopping Docker services...")
    docker_dir = project_root / "services" / "rag_services" / "docker"
    
    subprocess.run(
        ["docker-compose", "-f", "docker-compose.weaviate.yml", "down"],
        cwd=docker_dir,
        check=False,
        capture_output=True
    )
    subprocess.run(
        ["docker-compose", "-f", "docker-compose.opensearch.yml", "down"],
        cwd=docker_dir,
        check=False,
        capture_output=True
    )

def main():
    project_root = Path(__file__).parent.absolute()
    
    print("🛑 Stopping all Chatbot-UIT backend services...\n")
    
    # Kill Python services
    kill_port(8000)  # RAG Service
    kill_port(8001)  # Orchestrator
    
    # Stop Docker services
    stop_docker_services(project_root)
    
    print()
    print_success("All backend services stopped!")
    print_info("To start again, run: python start_backend.py")

if __name__ == "__main__":
    main()
