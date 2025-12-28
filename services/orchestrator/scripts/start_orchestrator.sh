#!/bin/bash

# Start Orchestrator Server with Multi-Agent System
# This script starts the FastAPI orchestrator server with proper environment setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Multi-Agent Orchestrator Server${NC}"
echo "=================================================="

# Check if we're in the correct directory
if [ ! -f "app/main.py" ]; then
    echo -e "${RED}❌ Error: Please run this script from the orchestrator directory${NC}"
    echo "   cd services/orchestrator"
    echo "   ./scripts/start_orchestrator.sh"
    exit 1
fi

# Check required environment variables
required_vars=("OPENROUTER_API_KEY")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo -e "${RED}❌ Missing required environment variables:${NC}"
    for var in "${missing_vars[@]}"; do
        echo "   $var"
    done
    echo
    echo "Please set these environment variables before starting the server."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required but not installed${NC}"
    exit 1
fi

# Check if uvicorn is available
if ! python3 -c "import uvicorn" &> /dev/null; then
    echo -e "${YELLOW}⚠️ Installing uvicorn...${NC}"
    pip install uvicorn[standard]
fi

# Set default environment variables if not set
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-"8001"}
export RELOAD=${RELOAD:-"true"}
export LOG_LEVEL=${LOG_LEVEL:-"info"}

# Multi-agent system configuration
export ENABLE_MULTI_AGENT=${ENABLE_MULTI_AGENT:-"true"}
export ENABLE_PLANNING=${ENABLE_PLANNING:-"true"}
export ENABLE_VERIFICATION=${ENABLE_VERIFICATION:-"true"}

echo -e "${GREEN}✅ Environment Configuration:${NC}"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Reload: $RELOAD"
echo "   Log Level: $LOG_LEVEL"
echo "   Multi-Agent: $ENABLE_MULTI_AGENT"
echo "   Planning: $ENABLE_PLANNING"
echo "   Verification: $ENABLE_VERIFICATION"
echo

# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo -e "${BLUE}🔧 Starting FastAPI server...${NC}"
echo "   Access the API at: http://$HOST:$PORT"
echo "   API Documentation: http://$HOST:$PORT/docs"
echo "   Health Check: http://$HOST:$PORT/health"
echo "   Agents Info: http://$HOST:$PORT/agents/info"
echo

# Start the server
if [ "$RELOAD" = "true" ]; then
    exec uvicorn app.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level "$LOG_LEVEL" \
        --access-log
else
    exec uvicorn app.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level "$LOG_LEVEL" \
        --access-log
fi