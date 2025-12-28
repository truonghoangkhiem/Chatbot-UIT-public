# Vietnamese Hybrid RAG System with Ports & Adapters Architecture
# Commands for managing the Clean Architecture implementation

.PHONY: help install start stop restart status clean test demo sample-data architecture \
        demo-clean test-arch migrate-check arch-info clean-arch start-legacy start-clean \
        validate cleanup-project info docs opensearch-create opensearch-delete opensearch-reset \
        opensearch-stats opensearch-health index-sample index-clear search-test perf-test \
        clean-cache logs logs-opensearch dev-run dev-shell test-vietnamese test-components test-full

# Default target
help:
	@echo "🚀 Vietnamese Hybrid RAG System - Ports & Adapters Architecture"
	@echo "================================================================"
	@echo ""
	@echo "🏗️  Architecture Commands:"
	@echo "  🎯 demo           - Demo Ports & Adapters architecture"
	@echo "  🧪 test-arch      - Test architecture components"
	@echo "  � arch-info      - Show architecture information"
	@echo "  � test-migration - Run comprehensive migration tests"
	@echo ""
	@echo "📦 Service Management:"
	@echo "  📦 install        - Install Python dependencies"
	@echo "  🚀 start          - Start all services (OpenSearch + RAG)"
	@echo "  ⏹️  stop           - Stop all services"
	@echo "  🔄 restart        - Restart all services"
	@echo "  📊 status         - Check service status"
	@echo "  🧹 clean          - Clean up containers and volumes"
	@echo ""
	@echo "🧪 Testing & Demo:"
	@echo "  📚 sample-data    - Create sample Vietnamese documents"
	@echo "  🧪 test-vietnamese - Test Vietnamese search capabilities"
	@echo "  🔬 test-full      - Run complete test pipeline"
	@echo ""
	@echo "📂 Index Management:"
	@echo "  📂 opensearch-*   - OpenSearch index management"
	@echo "  ❓ help           - Show this help message"

# Installation
install:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# Service management
start:
	@echo "🚀 Starting Vietnamese Hybrid RAG services..."
	@echo "🏗️  Architecture: Ports & Adapters (Clean Architecture)"
	@echo "📂 Checking OpenSearch..."
	@if curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; then \
		echo "✅ OpenSearch already running"; \
	else \
		echo "📂 Starting OpenSearch..."; \
		cd docker && docker-compose up -d opensearch || (echo "⚠️  Cleaning existing container..." && docker rm -f vietnamese-rag-opensearch 2>/dev/null && docker-compose up -d opensearch); \
		echo "⏳ Waiting for OpenSearch to be ready..."; \
		sleep 15; \
	fi
	@echo "🤖 Starting RAG service..."
	@if pgrep -f "uvicorn app.main:app" > /dev/null; then \
		echo "✅ RAG service already running"; \
	else \
		PYTHONPATH=. python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload & \
		echo "⏳ Waiting for RAG service to start..."; \
		sleep 5; \
	fi
	@echo "✅ All services started!"
	@echo "🌐 RAG API: http://localhost:8000"
	@echo "🔍 OpenSearch: http://localhost:9200"
	@echo "🏛️  Architecture docs: http://localhost:8000/docs"

start-dev:
	@echo "🚀 Starting development environment..."
	cd docker && docker-compose up -d
	@echo "⏳ Waiting for services to be ready..."
	@sleep 15
	@echo "✅ Development environment ready!"

stop:
	@echo "⏹️  Stopping services..."
	pkill -f "uvicorn app.main:app" || true
	cd docker && docker-compose down
	@echo "✅ All services stopped!"

restart: stop start
	@echo "🔄 Services restarted!"

status:
	@echo "📊 Service Status:"
	@echo "=================="
	@echo -n "🤖 RAG Service: "
	@curl -s http://localhost:8000/v1/health > /dev/null 2>&1 && echo "✅ Running" || echo "❌ Down"
	@echo -n "🔍 OpenSearch: "
	@curl -s http://localhost:9200 > /dev/null 2>&1 && echo "✅ Running" || echo "❌ Down"
	@echo -n "📦 Docker Services: "
	@cd docker && docker-compose ps --services --filter status=running | wc -l | xargs -I {} echo "{} running"

# Development commands
dev-run:
	@echo "🛠️  Running in development mode..."
	PYTHONPATH=. python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dev-shell:
	@echo "🐚 Opening development shell..."
	cd docker && docker-compose exec opensearch bash

# Sample data management
sample-data:
	@echo "📚 Creating sample Vietnamese data..."
	python scripts/create_sample_data.py

# Testing commands
test-vietnamese:
	@echo "🇻🇳 Testing Vietnamese search capabilities..."
	python scripts/test_vietnamese_search.py

test-components:
	@echo "🧪 Testing system components without Docker..."
	python scripts/test_without_docker.py

# Architecture demo
demo:
	@echo "🏗️  Running Ports & Adapters architecture demo..."
	python scripts/demo_clean_architecture.py

test-arch:
	@echo "🧪 Testing architecture components..."
	@echo "📦 Testing DI Container..."
	python -c "from core.container import get_container; c = get_container(); print('✅ Container OK')"
	@echo "🔍 Testing Core Domain..."
	python -c "from core.domain.search_service import SearchService; print('✅ Domain Service OK')"
	@echo "🔌 Testing Adapters..."
	python -c "from adapters.api_facade import get_search_facade; print('✅ API Facade OK')"
	@echo "✅ All architecture components working!"

test-migration:
	@echo "🧪 Running comprehensive migration tests..."
	python scripts/validate_migration_readiness.py

arch-info:
	@echo "🏗️  Architecture Information"
	@echo "============================="
	@echo "📚 Documentation:"
	@echo "   📖 Architecture Guide: PORTS_AND_ADAPTERS.md"
	@echo "   📋 Refactoring Summary: REFACTORING_SUMMARY.md"
	@echo ""
	@echo "🏛️  Architecture Layers:"
	@echo "   🎯 Core Domain: core/domain/"
	@echo "   🔌 Ports: core/ports/"
	@echo "   🏗️  Adapters: adapters/"
	@echo "   🌐 API Layer: app/api/"
	@echo ""
	@echo "🔧 Configuration:"
	@echo "   VECTOR_BACKEND=${VECTOR_BACKEND:-faiss}"
	@echo "   USE_HYBRID_SEARCH=${USE_HYBRID_SEARCH:-true}"
	@echo "   Architecture: Ports & Adapters (Clean Architecture)"

# Full test pipeline
test-full: sample-data test-vietnamese demo test-migration
	@echo "✅ Complete testing pipeline finished!"

# OpenSearch management
opensearch-create:
	@echo "📂 Creating OpenSearch index with Vietnamese analyzer..."
	python -c "import requests; r=requests.post('http://localhost:8000/v1/opensearch/create-index'); print(f'Status: {r.status_code}')"

opensearch-delete:
	@echo "🗑️  Deleting OpenSearch index..."
	python -c "import requests; r=requests.delete('http://localhost:8000/v1/opensearch/delete-index'); print(f'Status: {r.status_code}')"

opensearch-reset: opensearch-delete opensearch-create
	@echo "🔄 OpenSearch index reset complete!"

opensearch-stats:
	@echo "📊 Getting OpenSearch index statistics..."
	python -c "import requests; r=requests.get('http://localhost:8000/v1/opensearch/stats'); print(r.json() if r.status_code == 200 else f'Error: {r.status_code}')"

opensearch-health:
	@echo "🏥 Checking OpenSearch health..."
	python -c "import requests; r=requests.get('http://localhost:8000/v1/opensearch/health'); print(r.json() if r.status_code == 200 else f'Error: {r.status_code}')"

# Index management
index-sample:
	@echo "📝 Indexing sample documents..."
	python -c "from scripts.create_sample_data import index_sample_data; index_sample_data()"

index-clear:
	@echo "🧹 Clearing all indexed documents..."
	python -c "import requests; r=requests.delete('http://localhost:8000/v1/opensearch/clear'); print(f'Status: {r.status_code}')"

# Search testing
search-test:
	@echo "🔍 Running search tests..."
	python -c "import requests; r=requests.post('http://localhost:8000/v1/search', json={'query': 'tuyển sinh', 'search_mode': 'hybrid'}); print(f'Found: {len(r.json().get(\"hits\", []))} results')"

# Cleanup
clean:
	@echo "🧹 Cleaning up..."
	cd docker && docker-compose down -v --remove-orphans
	docker system prune -f
	@echo "✅ Cleanup complete!"

clean-cache:
	@echo "🧹 Cleaning Python cache..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "✅ Cache cleaned!"

clean-arch:
	@echo "🧹 Cleaning architecture artifacts..."
	@echo "🗑️  Removing Python cache..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "🗑️  Resetting DI container..."
	python -c "from core.container import reset_container; reset_container(); print('✅ Container reset')" 2>/dev/null || echo "ℹ️  Container not initialized"
	@echo "✅ Architecture cleanup complete!"

# Logs
logs:
	@echo "📋 Showing service logs..."
	cd docker && docker-compose logs -f

logs-opensearch:
	@echo "📋 Showing OpenSearch logs..."
	cd docker && docker-compose logs -f opensearch

# Performance testing
perf-test:
	@echo "⚡ Running performance tests..."
	python scripts/performance_test.py

# System info
info:
	@echo "ℹ️  System Information"
	@echo "====================="
	@echo "🐍 Python: $(shell python --version)"
	@echo "🐳 Docker: $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo "📁 Project: $(shell pwd)"
	@echo "🏗️  Architecture: Ports & Adapters (Clean Architecture)"
	@echo "📦 Key Dependencies:"
	@pip list | grep -E "(opensearch|faiss|sentence|transformers|llama-index)" || echo "   No key dependencies found"
	@echo ""
	@echo "📂 Project Structure:"
	@echo "   🎯 Core: $(shell find core -name "*.py" 2>/dev/null | wc -l) files"
	@echo "   🔌 Adapters: $(shell find adapters -name "*.py" 2>/dev/null | wc -l) files"
	@echo "   🌐 API: $(shell find app -name "*.py" 2>/dev/null | wc -l) files"
	@echo "   📝 Scripts: $(shell find scripts -name "*.py" 2>/dev/null | wc -l) files"

# Validation and cleanup
validate:
	@echo "✅ Validating project structure..."
	@echo "📂 Checking core architecture files..."
	@test -f core/container.py && echo "   ✅ DI Container: OK" || echo "   ❌ DI Container: Missing"
	@test -f core/domain/search_service.py && echo "   ✅ Domain Service: OK" || echo "   ❌ Domain Service: Missing"
	@test -f core/ports/repositories.py && echo "   ✅ Repository Ports: OK" || echo "   ❌ Repository Ports: Missing"
	@echo "🔌 Checking adapter implementations..."
	@test -f adapters/api_facade.py && echo "   ✅ API Facade: OK" || echo "   ❌ API Facade: Missing"
	@test -f adapters/llamaindex_vector_adapter.py && echo "   ✅ Vector Adapter: OK" || echo "   ❌ Vector Adapter: Missing"
	@echo "📚 Checking documentation..."
	@test -f PORTS_AND_ADAPTERS.md && echo "   ✅ Architecture Docs: OK" || echo "   ❌ Architecture Docs: Missing"
	@test -f REFACTORING_SUMMARY.md && echo "   ✅ Summary Docs: OK" || echo "   ❌ Summary Docs: Missing"
	@echo "✅ Validation complete!"

cleanup-project:
	@echo "🧹 Cleaning up project files..."
	@echo "🗑️  Removing Python cache..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "🗑️  Removing temporary files..."
	find . -name "*.tmp" -delete 2>/dev/null || true
	find . -name "*.log" -delete 2>/dev/null || true
	find . -name "*~" -delete 2>/dev/null || true
	@echo "🗑️  Removing OS specific files..."
	find . -name ".DS_Store" -delete 2>/dev/null || true
	find . -name "Thumbs.db" -delete 2>/dev/null || true
	@echo "✅ Project cleanup complete!"

# Documentation
docs:
	@echo "📚 Documentation Links"
	@echo "====================="
	@echo "🌐 API Docs: http://localhost:8000/docs"
	@echo "🔍 OpenSearch: http://localhost:9200"
	@echo "📊 Health Check: http://localhost:8000/v1/health"
	@echo "🧪 Test endpoint: make search-test"
	@echo ""
	@echo "🏗️  Architecture Documentation:"
	@echo "  📖 Ports & Adapters Guide: PORTS_AND_ADAPTERS.md"
	@echo "  📋 Refactoring Summary: REFACTORING_SUMMARY.md"
	@echo "  🎯 Demo Script: scripts/demo_clean_architecture.py"
	@echo ""
	@echo "🚀 Quick Start (Clean Architecture):"
	@echo "  1. make install"
	@echo "  2. make start-clean"
	@echo "  3. make sample-data"
	@echo "  4. make demo-clean"
	@echo ""
	@echo "🔧 Environment Variables:"
	@echo "  USE_CLEAN_ARCHITECTURE=true|false (default: true)"
	@echo "  VECTOR_BACKEND=faiss|chroma (default: faiss)"
	@echo "  USE_HYBRID_SEARCH=true|false (default: true)"
