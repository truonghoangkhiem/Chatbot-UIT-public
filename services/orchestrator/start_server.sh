#!/bin/bash

# Start script for orchestrator service

echo "🚀 Starting Chatbot-UIT Orchestrator Service"
echo "============================================"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from example..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration"
    echo "   Especially set OPENROUTER_API_KEY"
fi

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

# Check required environment variables
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ OPENROUTER_API_KEY is required"
    echo "   Please set this in your .env file"
    exit 1
fi

# Install dependencies if requirements.txt changed
if [ requirements.txt -nt venv/lib/python*/site-packages/installed.flag ] 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    touch venv/lib/python*/site-packages/installed.flag 2>/dev/null || true
fi

# Default values
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8002}
LOG_LEVEL=${LOG_LEVEL:-"info"}

echo "🌐 Server will start on: http://${HOST}:${PORT}"
echo "📚 API documentation: http://${HOST}:${PORT}/docs"
echo "💚 Health check: http://${HOST}:${PORT}/api/v1/health"
echo ""

# Start the server
uvicorn app.main:app --host $HOST --port $PORT --log-level $LOG_LEVEL --reload