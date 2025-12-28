#!/bin/bash
# scripts/start_hybrid_system.sh
#
# Description:
# Complete startup script for Hybrid RAG System.
# Starts OpenSearch, syncs data, and launches the RAG service.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAG_SERVICE_DIR="$PROJECT_ROOT/services/rag_services"
DOCKER_DIR="$RAG_SERVICE_DIR/docker"

echo -e "${BLUE}🚀 Starting Hybrid RAG System${NC}"
echo "=================================="

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for service
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    echo "Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            print_status "$service_name is ready!"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts: Waiting for $service_name..."
        sleep 2
        ((attempt++))
    done
    
    print_error "$service_name failed to start within expected time"
    return 1
}

# Check prerequisites
echo -e "${BLUE}📋 Checking Prerequisites${NC}"
echo "----------------------------"

if ! command_exists docker; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi
print_status "Docker found"

if ! command_exists docker-compose; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi
print_status "Docker Compose found"

if ! command_exists python3; then
    print_error "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi
print_status "Python 3 found"

# Check if we're in the right directory
if [ ! -d "$RAG_SERVICE_DIR" ]; then
    print_error "RAG service directory not found: $RAG_SERVICE_DIR"
    exit 1
fi
print_status "RAG service directory found"

# Change to RAG service directory
cd "$RAG_SERVICE_DIR"

# Check Python dependencies
echo -e "\n${BLUE}📦 Checking Python Dependencies${NC}"
echo "-----------------------------------"

if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
print_status "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Start OpenSearch
echo -e "\n${BLUE}🔍 Starting OpenSearch${NC}"
echo "-------------------------"

cd "$DOCKER_DIR"

# Check if OpenSearch is already running
if docker-compose -f docker-compose.opensearch.yml ps | grep -q "Up"; then
    print_warning "OpenSearch containers are already running"
else
    print_status "Starting OpenSearch containers..."
    docker-compose -f docker-compose.opensearch.yml up -d
fi

# Wait for OpenSearch to be ready
if wait_for_service "http://localhost:9200/_cluster/health" "OpenSearch"; then
    print_status "OpenSearch cluster is healthy"
else
    print_error "Failed to start OpenSearch"
    exit 1
fi

# Wait for OpenSearch Dashboards (optional)
if wait_for_service "http://localhost:5601/api/status" "OpenSearch Dashboards"; then
    print_status "OpenSearch Dashboards is ready"
else
    print_warning "OpenSearch Dashboards may not be fully ready (this is optional)"
fi

# Back to RAG service directory
cd "$RAG_SERVICE_DIR"

# Check if .env file exists
echo -e "\n${BLUE}⚙️  Checking Configuration${NC}"
echo "-----------------------------"

if [ ! -f ".env" ]; then
    print_warning "Creating default .env file..."
    cat > .env << EOF
# OpenSearch Configuration
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_INDEX=rag_documents
OPENSEARCH_USERNAME=admin
OPENSEARCH_PASSWORD=admin
OPENSEARCH_USE_SSL=false
OPENSEARCH_VERIFY_CERTS=false

# Hybrid Search Settings
USE_HYBRID_SEARCH=true
BM25_WEIGHT=0.5
VECTOR_WEIGHT=0.5
RRF_RANK_CONSTANT=60

# Model Configuration
EMB_MODEL=intfloat/multilingual-e5-base
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
VECTOR_BACKEND=faiss
STORAGE_DIR=./storage

# Application Settings
APP_ENV=dev
PORT=8000
LOG_LEVEL=INFO
EOF
    print_status ".env file created with default settings"
else
    print_status ".env file already exists"
fi

# Test OpenSearch connection
echo -e "\n${BLUE}🔧 Testing System Components${NC}"
echo "--------------------------------"

print_status "Testing OpenSearch connection..."
python -c "
import sys
sys.path.append('.')
try:
    from store.opensearch.client import get_opensearch_client
    client = get_opensearch_client()
    if client.health_check():
        print('✅ OpenSearch connection successful')
    else:
        print('❌ OpenSearch connection failed')
        sys.exit(1)
except Exception as e:
    print(f'❌ OpenSearch connection error: {e}')
    sys.exit(1)
"

# Check if we have any documents to sync
echo -e "\n${BLUE}📚 Syncing Documents${NC}"
echo "--------------------"

# Check if vector index has documents
python -c "
import sys
sys.path.append('.')
try:
    from retrieval.engine import _load_or_create_index, _ensure_settings
    _ensure_settings()
    index = _load_or_create_index()
    docstore = index.storage_context.docstore
    doc_count = len(docstore.docs)
    print(f'Vector store contains {doc_count} documents')
    if doc_count == 0:
        print('⚠️  No documents found in vector store. Please add documents first.')
    else:
        print('📊 Proceeding with document sync to OpenSearch...')
except Exception as e:
    print(f'⚠️  Could not check vector store: {e}')
"

# Sync documents to OpenSearch
if [ -f "scripts/sync_to_opensearch.py" ]; then
    print_status "Syncing documents to OpenSearch..."
    python scripts/sync_to_opensearch.py
else
    print_warning "Sync script not found, skipping document sync"
fi

# Start RAG service
echo -e "\n${BLUE}🌐 Starting RAG Service${NC}"
echo "-------------------------"

print_status "Starting RAG service on port 8000..."
echo ""
echo -e "${GREEN}🎉 System is starting up!${NC}"
echo ""
echo "Services:"
echo "  🔍 OpenSearch:         http://localhost:9200"
echo "  📊 OpenSearch Dashboards: http://localhost:5601"
echo "  🤖 RAG API:           http://localhost:8000"
echo "  📚 API Docs:          http://localhost:8000/docs"
echo ""
echo "Useful endpoints:"
echo "  Health:    GET  /v1/opensearch/health"
echo "  Search:    POST /v1/search"
echo "  BM25:      POST /v1/opensearch/search"
echo "  Stats:     GET  /v1/opensearch/stats"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Start the RAG service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
