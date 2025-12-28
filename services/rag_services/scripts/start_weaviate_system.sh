#!/bin/bash

# Start Weaviate RAG System
# This script starts all necessary services for the Vietnamese RAG system with Weaviate

set -e

echo "🚀 Starting Vietnamese RAG System with Weaviate..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo -e "${BLUE}Step 1: Starting Docker services...${NC}"
cd docker
docker-compose up -d

# Wait for services to be healthy
echo ""
echo -e "${BLUE}Step 2: Waiting for services to be ready...${NC}"

# Wait for Weaviate
echo -n "Waiting for Weaviate..."
for i in {1..30}; do
    if curl -s http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

# Wait for OpenSearch
echo -n "Waiting for OpenSearch..."
for i in {1..30}; do
    if curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

cd ..

echo ""
echo -e "${GREEN}✅ All services are ready!${NC}"
echo ""
echo "Service endpoints:"
echo "  - Weaviate:   http://localhost:8080"
echo "  - OpenSearch: http://localhost:9200"
echo ""

# Check if Python environment is set up
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found${NC}"
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo ""
echo -e "${BLUE}Step 3: Starting RAG service...${NC}"
echo ""

# Start the FastAPI server
python start_server.py

# Cleanup on exit
trap "echo 'Stopping services...'; cd docker; docker-compose down; cd .." EXIT
