#!/bin/bash
# Start Neo4j Graph Web Viewer

echo "======================================================================"
echo "🌐 Starting Neo4j Graph Web Viewer"
echo "======================================================================"
echo ""

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 Installing Flask..."
    pip install flask >/dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ Flask installed successfully"
    else
        echo "❌ Failed to install Flask"
        echo "Please run: pip install flask"
        exit 1
    fi
else
    echo "✅ Flask is already installed"
fi

echo ""
echo "🚀 Starting web server..."
echo ""
echo "======================================================================"
echo "📍 OPEN YOUR BROWSER: http://localhost:5555"
echo "======================================================================"
echo ""
echo "Features:"
echo "  ✅ Proper UTF-8 encoding for Vietnamese"
echo "  ✅ View all nodes and relationships"
echo "  ✅ Run custom Cypher queries"
echo "  ✅ Beautiful UI with statistics"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 scripts/web_graph_viewer.py
