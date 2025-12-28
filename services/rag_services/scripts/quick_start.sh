#!/bin/bash
# scripts/quick_start.sh
#
# Quick start script for Vietnamese Hybrid RAG System
# Automatically sets up and demos the complete system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo
    print_color $CYAN "=================================="
    print_color $CYAN "$1"  
    print_color $CYAN "=================================="
}

print_step() {
    print_color $BLUE "📋 $1"
}

print_success() {
    print_color $GREEN "✅ $1"
}

print_warning() {
    print_color $YELLOW "⚠️  $1"
}

print_error() {
    print_color $RED "❌ $1"
}

# Check if required tools are installed
check_requirements() {
    print_step "Checking requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is required but not installed"
        print_warning "Please install Docker and Docker Compose"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is required but not installed"
        exit 1
    fi
    
    print_success "All requirements satisfied"
}

# Install Python dependencies
install_dependencies() {
    print_step "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Start services
start_services() {
    print_step "Starting services..."
    
    # Start OpenSearch
    print_color $BLUE "🔍 Starting OpenSearch..."
    cd docker && docker-compose up -d opensearch
    cd ..
    
    # Wait for OpenSearch
    print_step "Waiting for OpenSearch to be ready..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:9200/_cluster/health &> /dev/null; then
            print_success "OpenSearch is ready"
            break
        fi
        
        printf "."
        sleep 2
        ((attempt++))
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_error "OpenSearch failed to start within 60 seconds"
        exit 1
    fi
    
    # Start RAG service in background
    print_color $BLUE "🤖 Starting RAG service..."
    python app/main.py &
    RAG_PID=$!
    
    # Wait for RAG service
    print_step "Waiting for RAG service to be ready..."
    attempt=0
    max_attempts=15
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:8000/v1/health &> /dev/null; then
            print_success "RAG service is ready"
            break
        fi
        
        printf "."
        sleep 2
        ((attempt++))
    done
    
    if [ $attempt -eq $max_attempts ]; then
        print_error "RAG service failed to start within 30 seconds"
        kill $RAG_PID 2>/dev/null || true
        exit 1
    fi
}

# Create OpenSearch index
create_index() {
    print_step "Creating OpenSearch index with Vietnamese analyzer..."
    
    local response=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/v1/opensearch/create-index)
    
    if [ "$response" = "200" ] || [ "$response" = "201" ]; then
        print_success "OpenSearch index created"
    else
        print_warning "Index creation returned status: $response (might already exist)"
    fi
}

# Create and index sample data
setup_sample_data() {
    print_step "Creating sample Vietnamese documents..."
    
    if python scripts/create_sample_data.py; then
        print_success "Sample data created and indexed"
    else
        print_error "Failed to create sample data"
        exit 1
    fi
}

# Run tests
run_tests() {
    print_step "Running Vietnamese search tests..."
    
    if python scripts/test_vietnamese_search.py; then
        print_success "Vietnamese tests passed"
    else
        print_warning "Some Vietnamese tests may have failed"
    fi
}

# Run demo
run_demo() {
    print_step "Running complete system demo..."
    
    if python scripts/demo_hybrid_search.py; then
        print_success "Demo completed successfully"
    else
        print_error "Demo failed"
        return 1
    fi
}

# Show system information
show_info() {
    print_header "SYSTEM INFORMATION"
    
    echo "🌐 Services:"
    echo "  • RAG API: http://localhost:8000"
    echo "  • OpenSearch: http://localhost:9200"
    echo "  • API Docs: http://localhost:8000/docs"
    echo
    
    echo "📊 Index Statistics:"
    local stats=$(curl -s http://localhost:8000/v1/opensearch/stats 2>/dev/null || echo "{}")
    local doc_count=$(echo $stats | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('total_documents', 'Unknown'))" 2>/dev/null || echo "Unknown")
    echo "  • Total documents: $doc_count"
    echo
    
    echo "🛠️  Management Commands:"
    echo "  • make status      - Check service status"
    echo "  • make stop        - Stop all services"
    echo "  • make demo        - Run demo again"
    echo "  • make help        - Show all commands"
}

# Cleanup on exit
cleanup() {
    if [ ! -z "$RAG_PID" ]; then
        print_step "Cleaning up RAG service..."
        kill $RAG_PID 2>/dev/null || true
    fi
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    print_header "🚀 VIETNAMESE HYBRID RAG QUICK START"
    
    print_color $YELLOW "This script will:"
    echo "  1. ✅ Check requirements"
    echo "  2. 📦 Install dependencies"  
    echo "  3. 🚀 Start services (OpenSearch + RAG)"
    echo "  4. 📂 Create index with Vietnamese analyzer"
    echo "  5. 📚 Create sample Vietnamese documents"
    echo "  6. 🧪 Run Vietnamese search tests"
    echo "  7. 🎯 Run complete system demo"
    echo
    
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Setup cancelled by user"
        exit 0
    fi
    
    # Execute setup steps
    check_requirements
    install_dependencies
    start_services
    create_index
    setup_sample_data
    run_tests
    
    print_header "✅ SETUP COMPLETED SUCCESSFULLY!"
    show_info
    
    echo
    print_color $GREEN "🎉 Vietnamese Hybrid RAG System is ready!"
    echo
    print_color $CYAN "Would you like to run the interactive demo? (y/N): "
    read -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_demo
    else
        print_color $YELLOW "Demo skipped. You can run it later with: make demo"
    fi
    
    echo
    print_color $GREEN "🚀 System is running! Press Ctrl+C to stop services."
    
    # Keep script running to maintain services
    echo "⏳ Keeping services running... (Press Ctrl+C to stop)"
    
    # Wait for interrupt
    while true; do
        sleep 5
        # Check if services are still running
        if ! curl -s http://localhost:8000/v1/health &> /dev/null; then
            print_error "RAG service appears to have stopped"
            break
        fi
        
        if ! curl -s http://localhost:9200/_cluster/health &> /dev/null; then
            print_error "OpenSearch appears to have stopped"
            break
        fi
    done
}

# Check if we're in the right directory
if [ ! -f "app/main.py" ]; then
    print_error "Please run this script from the rag_services directory"
    print_warning "Expected directory structure:"
    print_warning "  rag_services/"
    print_warning "    app/main.py"
    print_warning "    requirements.txt"
    print_warning "    scripts/"
    exit 1
fi

# Run main function
main "$@"
