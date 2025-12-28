#!/usr/bin/env python3
"""
Backend Startup Script for Chatbot-UIT
======================================
Khởi động toàn bộ backend services:
- Docker services (OpenSearch, Weaviate)
- RAG Service (port 8000)
- Orchestrator Service (port 8001)

Usage:
    python start_backend.py
    python start_backend.py --skip-docker  # Skip Docker services
    python start_backend.py --stop         # Stop all services
"""

import subprocess
import sys
import time
import signal
import os
import argparse
from pathlib import Path
from typing import List, Optional
import json

# Colors for terminal output
class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color
    BOLD = '\033[1m'

# Global variables for process management
processes: List[subprocess.Popen] = []
docker_services_started = False

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
    print(f"{Colors.BLUE}{text.center(70)}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*70}{Colors.NC}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.NC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.YELLOW}ℹ {text}{Colors.NC}")

def print_step(step: str, total: str, text: str):
    """Print step message"""
    print(f"\n{Colors.BOLD}[{step}/{total}] {text}{Colors.NC}")

def run_command(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        if check:
            print_error(f"Command failed: {' '.join(cmd)}")
            print_error(f"Error: {e.stderr}")
            raise
        return e

def check_port(port: int) -> bool:
    """Check if a port is in use"""
    result = run_command(
        ["lsof", "-Pi", f":{port}", "-sTCP:LISTEN", "-t"],
        check=False
    )
    return result.returncode == 0

def kill_port(port: int):
    """Kill process on a specific port"""
    print_info(f"Killing process on port {port}...")
    run_command(
        ["bash", "-c", f"lsof -ti:{port} | xargs kill -9 2>/dev/null || true"],
        check=False
    )
    time.sleep(1)

def check_docker() -> bool:
    """Check if Docker is running"""
    result = run_command(["docker", "ps"], check=False)
    return result.returncode == 0

def check_conda_env() -> bool:
    """Check if conda environment 'chatbot-UIT' exists and is activated"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env == 'chatbot-UIT':
        return True
    
    # Check if environment exists
    result = run_command(
        ["conda", "env", "list"],
        check=False
    )
    if result.returncode == 0 and 'chatbot-UIT' in result.stdout:
        print_info("Conda environment 'chatbot-UIT' exists but not activated")
        return True
    return False

def start_docker_services(project_root: Path):
    """Start Docker services (OpenSearch, Weaviate, and Neo4j)"""
    global docker_services_started
    
    print_step("1", "4", "Starting Docker Services")
    
    if not check_docker():
        print_error("Docker is not running!")
        print_info("Please start Docker Desktop or Docker daemon first")
        sys.exit(1)
    
    docker_dir = project_root / "services" / "rag_services" / "docker"
    
    # Check if services are already running
    if check_port(9200) and check_port(8090) and check_port(7687):
        print_success("Docker services already running")
        print_info("  - OpenSearch: http://localhost:9200")
        print_info("  - Weaviate: http://localhost:8090")
        print_info("  - Neo4j: bolt://localhost:7687")
        docker_services_started = True
        return
    
    # Stop existing services to recreate network
    print_info("Stopping existing Docker services...")
    run_command(
        ["docker-compose", "-f", "docker-compose.weaviate.yml", "down"],
        cwd=docker_dir,
        check=False
    )
    run_command(
        ["docker-compose", "-f", "docker-compose.opensearch.yml", "down"],
        cwd=docker_dir,
        check=False
    )
    run_command(
        ["docker-compose", "-f", "docker-compose.neo4j.yml", "down"],
        cwd=docker_dir,
        check=False
    )
    time.sleep(2)
    
    # Start all services together to share network
    print_info("Starting OpenSearch, Weaviate, and Neo4j...")
    result = run_command(
        [
            "docker-compose",
            "-f", "docker-compose.opensearch.yml",
            "-f", "docker-compose.weaviate.yml",
            "-f", "docker-compose.neo4j.yml",
            "up", "-d"
        ],
        cwd=docker_dir
    )
    
    if result.returncode == 0:
        print_success("Docker services started successfully")
        docker_services_started = True
        
        # Wait for services to be healthy
        print_info("Waiting for services to be ready...")
        max_retries = 30
        for i in range(max_retries):
            opensearch_healthy = run_command(
                ["curl", "-sf", "http://localhost:9200/_cluster/health"],
                check=False
            ).returncode == 0
            
            weaviate_healthy = run_command(
                ["curl", "-sf", "http://localhost:8090/v1/.well-known/ready"],
                check=False
            ).returncode == 0
            
            neo4j_healthy = run_command(
                ["curl", "-sf", "http://localhost:7474"],
                check=False
            ).returncode == 0
            
            if opensearch_healthy and weaviate_healthy and neo4j_healthy:
                print_success("All Docker services are healthy!")
                break
            
            time.sleep(2)
            print(".", end="", flush=True)
        else:
            print_error("\nServices did not become healthy in time")
            print_info("They may still be starting up. Check with: docker ps")
        
        print()
        print_info("Docker services:")
        print_info("  - OpenSearch: http://localhost:9200")
        print_info("  - OpenSearch Dashboards: http://localhost:5601")
        print_info("  - Weaviate: http://localhost:8090")
        print_info("  - Neo4j Browser: http://localhost:7474 (neo4j/[NEO4J_PASSWORD])")
        print_info("  - Neo4j Bolt: bolt://localhost:7687")
    else:
        print_error("Failed to start Docker services")
        sys.exit(1)

def start_rag_service(project_root: Path, debug_mode: bool = False):
    """Start RAG Service"""
    global processes
    
    print_step("2", "4", "Starting RAG Service")
    
    # Check if already running
    if check_port(8000):
        print_info("Port 8000 is already in use")
        response = input("Kill existing process and restart? (y/n): ")
        if response.lower() == 'y':
            kill_port(8000)
        else:
            print_info("Skipping RAG service start")
            return
    
    rag_dir = project_root / "services" / "rag_services"
    
    # Check if conda environment is activated
    python_exe = sys.executable
    if 'chatbot-UIT' not in python_exe:
        print_error("Conda environment 'chatbot-UIT' is not activated!")
        print_info("Please run: conda activate chatbot-UIT")
        print_info("Then run this script again")
        sys.exit(1)
    
    print_info(f"Using Python: {python_exe}")
    print_info("Starting RAG Service on port 8000...")
    if debug_mode:
        print_info("🐛 DEBUG MODE ENABLED - Detailed logging")
    print_info("Logs will appear below (Ctrl+C to stop all services)...")
    print()
    
    # Set environment variable for debug mode
    env = os.environ.copy()
    if debug_mode:
        env['LOG_LEVEL'] = 'DEBUG'
    
    # Start RAG service - no output capture, show logs in terminal
    proc = subprocess.Popen(
        [python_exe, "start_server.py"],
        cwd=rag_dir,
        env=env
    )
    processes.append(proc)
    
    # Wait for service to be ready
    print_info("Waiting for RAG service to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            result = run_command(
                ["curl", "-sf", "http://localhost:8000/v1/health"],
                check=False
            )
            if result.returncode == 0:
                print_success("RAG Service is ready!")
                print_info("  - API: http://localhost:8000")
                print_info("  - Docs: http://localhost:8000/docs")
                print_info("  - Health: http://localhost:8000/v1/health")
                return
        except:
            pass
        
        time.sleep(1)
        print(".", end="", flush=True)
    
    print_error("\nRAG Service did not start properly")
    print_info("Check logs manually or try running:")
    print_info(f"  cd {rag_dir}")
    print_info("  python start_server.py")
    sys.exit(1)

def start_orchestrator_service(project_root: Path, debug_mode: bool = False):
    """Start Orchestrator Service"""
    global processes
    
    print_step("3", "4", "Starting Orchestrator Service")
    
    # Check if already running
    if check_port(8001):
        print_info("Port 8001 is already in use")
        response = input("Kill existing process and restart? (y/n): ")
        if response.lower() == 'y':
            kill_port(8001)
        else:
            print_info("Skipping Orchestrator service start")
            return
    
    orchestrator_dir = project_root / "services" / "orchestrator"
    
    # Check if conda environment is activated
    python_exe = sys.executable
    if 'chatbot-UIT' not in python_exe:
        print_error("Conda environment 'chatbot-UIT' is not activated!")
        sys.exit(1)
    
    print_info(f"Using Python: {python_exe}")
    print_info("Starting Orchestrator Service on port 8001...")
    if debug_mode:
        print_info("🐛 DEBUG MODE ENABLED - Detailed agent I/O logging")
    print()
    
    # Load environment variables
    env = os.environ.copy()
    env_file = orchestrator_dir / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        env[key.strip()] = value.strip()
    
    # Set debug log level if requested
    if debug_mode:
        env['LOG_LEVEL'] = 'DEBUG'
    
    # Disable timeout for Answer Agent (deepseek model can be slow)
    env['OPENROUTER_TIMEOUT'] = 'none'
    
    # Add rag_services to PYTHONPATH for Graph Reasoning (Neo4j adapter)
    rag_services_path = str(orchestrator_dir.parent / "rag_services")
    existing_pythonpath = env.get('PYTHONPATH', '')
    if existing_pythonpath:
        env['PYTHONPATH'] = f"{orchestrator_dir}:{rag_services_path}:{existing_pythonpath}"
    else:
        env['PYTHONPATH'] = f"{orchestrator_dir}:{rag_services_path}"
    
    print_info(f"PYTHONPATH set for Graph Reasoning: {env['PYTHONPATH'][:100]}...")
    
    # Start Orchestrator service - no output capture, show logs in terminal
    proc = subprocess.Popen(
        [
            python_exe, "-m", "uvicorn",
            "app.main:app",
            "--host", "0.0.0.0",
            "--port", "8001",
            "--log-level", "info"
        ],
        cwd=orchestrator_dir,
        env=env
    )
    processes.append(proc)
    
    # Wait for service to be ready
    print_info("Waiting for Orchestrator to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            result = run_command(
                ["curl", "-sf", "http://localhost:8001/api/v1/health"],
                check=False
            )
            if result.returncode == 0:
                print_success("Orchestrator Service is ready!")
                print_info("  - API: http://localhost:8001")
                print_info("  - Docs: http://localhost:8001/docs")
                print_info("  - Health: http://localhost:8001/api/v1/health")
                return
        except:
            pass
        
        time.sleep(1)
        print(".", end="", flush=True)
    
    print_error("\nOrchestrator Service did not start properly")
    sys.exit(1)

def print_summary():
    """Print summary of running services"""
    print_step("4", "4", "Backend Services Summary")
    
    print(f"\n{Colors.GREEN}{'='*70}{Colors.NC}")
    print(f"{Colors.GREEN}{'🎉 All Backend Services Started Successfully!'.center(70)}{Colors.NC}")
    print(f"{Colors.GREEN}{'='*70}{Colors.NC}\n")
    
    print(f"{Colors.BOLD}Service URLs:{Colors.NC}")
    print(f"  {Colors.GREEN}✓{Colors.NC} RAG Service:          http://localhost:8000")
    print(f"  {Colors.GREEN}✓{Colors.NC} Orchestrator:         http://localhost:8001")
    print(f"  {Colors.GREEN}✓{Colors.NC} OpenSearch:           http://localhost:9200")
    print(f"  {Colors.GREEN}✓{Colors.NC} OpenSearch Dashboards: http://localhost:5601")
    print(f"  {Colors.GREEN}✓{Colors.NC} Weaviate:             http://localhost:8090")
    print(f"  {Colors.GREEN}✓{Colors.NC} Neo4j Browser:        http://localhost:7474")
    print(f"  {Colors.GREEN}✓{Colors.NC} Neo4j Bolt:           bolt://localhost:7687")
    
    print(f"\n{Colors.BOLD}API Documentation:{Colors.NC}")
    print(f"  - RAG API Docs:         http://localhost:8000/docs")
    print(f"  - Orchestrator Docs:    http://localhost:8001/docs")
    
    print(f"\n{Colors.BOLD}Database Credentials:{Colors.NC}")
    print(f"  - Neo4j: neo4j / [NEO4J_PASSWORD from .env]")
    print(f"  - OpenSearch: admin / admin")
    
    print(f"\n{Colors.BOLD}Health Checks:{Colors.NC}")
    print(f"  curl http://localhost:8000/v1/health")
    print(f"  curl http://localhost:8001/api/v1/health")
    
    print(f"\n{Colors.BOLD}Testing:{Colors.NC}")
    print(f"  python services/orchestrator/tests/demo_agent_rag.py")
    
    print(f"\n{Colors.BOLD}{Colors.YELLOW}{'─'*70}{Colors.NC}")
    print(f"{Colors.YELLOW}  Logs đang hiển thị real-time bên dưới...{Colors.NC}")
    print(f"{Colors.YELLOW}  Nhấn Ctrl+C để dừng tất cả services{Colors.NC}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{'─'*70}{Colors.NC}\n")

def stop_all_services(project_root: Path):
    """Stop all running services"""
    print_header("Stopping All Backend Services")
    
    # Stop Python processes
    for proc in processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except:
            proc.kill()
    
    # Kill processes on ports
    print_info("Stopping services on ports...")
    kill_port(8000)
    kill_port(8001)
    
    # Stop Docker services
    if docker_services_started:
        print_info("Stopping Docker services...")
        docker_dir = project_root / "services" / "rag_services" / "docker"
        
        run_command(
            ["docker-compose", "-f", "docker-compose.weaviate.yml", "down"],
            cwd=docker_dir,
            check=False
        )
        run_command(
            ["docker-compose", "-f", "docker-compose.opensearch.yml", "down"],
            cwd=docker_dir,
            check=False
        )
        run_command(
            ["docker-compose", "-f", "docker-compose.neo4j.yml", "down"],
            cwd=docker_dir,
            check=False
        )
    
    print_success("All services stopped")

def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    print(f"\n\n{Colors.YELLOW}Received interrupt signal. Stopping services...{Colors.NC}")
    project_root = Path(__file__).parent.absolute()
    stop_all_services(project_root)
    sys.exit(0)

def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Start Chatbot-UIT Backend Services')
    parser.add_argument('--skip-docker', action='store_true', help='Skip Docker services startup')
    parser.add_argument('--stop', action='store_true', help='Stop all services')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug logging (debug is enabled by default)')
    args = parser.parse_args()
    
    # Debug mode is ON by default, use --no-debug to turn it off
    debug_mode = not args.no_debug
    
    # Get project root
    project_root = Path(__file__).parent.absolute()
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Stop services if requested
    if args.stop:
        stop_all_services(project_root)
        return
    
    # Print header
    print_header("🚀 Starting Chatbot-UIT Backend Services")
    
    print_info(f"Project root: {project_root}")
    print_info(f"Python: {sys.executable}")
    
    # Show debug mode status
    if debug_mode:
        print_success("🐛 Debug mode: ENABLED (use --no-debug to disable)")
    else:
        print_info("Debug mode: DISABLED (default is enabled)")
    
    # Check conda environment
    if not check_conda_env():
        print_error("Conda environment 'chatbot-UIT' not found!")
        print_info("Please create and activate it first:")
        print_info("  conda create -n chatbot-UIT python=3.11")
        print_info("  conda activate chatbot-UIT")
        sys.exit(1)
    
    if os.environ.get('CONDA_DEFAULT_ENV') != 'chatbot-UIT':
        print_error("Conda environment 'chatbot-UIT' is not activated!")
        print_info("Please run: conda activate chatbot-UIT")
        print_info("Then run this script again")
        sys.exit(1)
    
    try:
        # Start Docker services
        if not args.skip_docker:
            start_docker_services(project_root)
        else:
            print_info("Skipping Docker services (--skip-docker)")
        
        # Start RAG service
        start_rag_service(project_root, debug_mode=debug_mode)
        
        # Start Orchestrator service
        start_orchestrator_service(project_root, debug_mode=debug_mode)
        
        # Print summary
        print_summary()
        
        # Keep script running - wait for any process to finish or Ctrl+C
        print_info("Services running. Monitoring processes...")
        try:
            # Wait for all processes to finish (they won't unless error)
            for proc in processes:
                proc.wait()
        except KeyboardInterrupt:
            # This will be caught by outer handler
            raise
    
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.NC}")
        stop_all_services(project_root)
    except Exception as e:
        print_error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        stop_all_services(project_root)
        sys.exit(1)

if __name__ == "__main__":
    main()
