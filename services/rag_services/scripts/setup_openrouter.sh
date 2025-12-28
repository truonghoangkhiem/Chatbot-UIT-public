#!/bin/bash
# Quick setup script for OpenRouter configuration

echo "=============================================================================="
echo "OPENROUTER SETUP HELPER"
echo "=============================================================================="
echo ""

# Check if .env exists
if [ -f .env ]; then
    echo "✅ Found .env file"
else
    echo "❌ .env file not found!"
    echo "Creating from .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Created .env from .env.example"
    else
        echo "❌ .env.example not found either!"
        exit 1
    fi
fi

echo ""
echo "Current LLM Configuration:"
echo "-------------------------"
grep "LLM_PROVIDER" .env || echo "LLM_PROVIDER not set"
grep "LLM_MODEL" .env || echo "LLM_MODEL not set"
grep "OPENAI_API_KEY" .env || echo "OPENAI_API_KEY not set"
grep "OPENAI_BASE_URL" .env || echo "OPENAI_BASE_URL not set"

echo ""
echo "=============================================================================="
echo "OPENROUTER API KEY CHECK"
echo "=============================================================================="

# Check if API key is set
API_KEY=$(grep "^OPENAI_API_KEY=" .env | cut -d'=' -f2 | xargs)

if [ -z "$API_KEY" ]; then
    echo "❌ OPENAI_API_KEY is NOT set!"
    echo ""
    echo "📝 To fix this:"
    echo "   1. Get your API key from: https://openrouter.ai/keys"
    echo "   2. Edit .env file:"
    echo "      OPENAI_API_KEY=sk-or-v1-YOUR_KEY_HERE"
    echo ""
    echo "Or run this command:"
    echo "   read -p 'Enter your OpenRouter API key: ' KEY && sed -i \"s/^OPENAI_API_KEY=.*/OPENAI_API_KEY=\$KEY/\" .env"
    echo ""
    exit 1
else
    # Check if it's a placeholder
    if [[ "$API_KEY" == *"YOUR_KEY_HERE"* ]] || [[ "$API_KEY" == *"sk-or-v1-..."* ]]; then
        echo "❌ OPENAI_API_KEY is still a placeholder!"
        echo ""
        echo "Current value: $API_KEY"
        echo ""
        echo "Please replace with your actual OpenRouter API key"
        exit 1
    else
        echo "✅ OPENAI_API_KEY is set!"
        echo "   Key: ${API_KEY:0:20}..."
    fi
fi

echo ""
echo "=============================================================================="
echo "NEO4J CONNECTION CHECK"
echo "=============================================================================="

# Check if Neo4j is running
if docker ps | grep -q neo4j; then
    echo "✅ Neo4j container is running"
    NEO4J_CONTAINER=$(docker ps --filter "ancestor=neo4j" --format "{{.Names}}" | head -1)
    echo "   Container: $NEO4J_CONTAINER"
else
    echo "❌ Neo4j container is NOT running"
    echo ""
    echo "Start Neo4j with:"
    echo "   docker-compose -f docker/docker-compose.neo4j.yml up -d"
    echo ""
fi

echo ""
echo "=============================================================================="
echo "READY TO RUN"
echo "=============================================================================="
echo ""
echo "✅ Configuration looks good!"
echo ""
echo "Next steps:"
echo ""
echo "1. Test extraction:"
echo "   python scripts/demo_openrouter_extraction.py"
echo ""
echo "2. Check system status:"
echo "   python scripts/test_graph_status.py"
echo ""
echo "3. Build graph from indexed data:"
echo "   python scripts/build_graph_from_indexed_data.py"
echo ""
echo "4. View graph in Neo4j Browser:"
echo "   http://localhost:7474"
echo "   Login: neo4j / [your NEO4J_PASSWORD]"
echo ""
