#!/bin/bash

# Chatbot UIT Frontend - Development Server
# This script starts the frontend development server

echo "🚀 Starting Chatbot UIT Frontend..."
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    echo ""
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from .env.example..."
    cp .env.example .env
    echo ""
fi

echo "✅ Starting development server..."
echo "📱 Frontend will be available at: http://localhost:5173"
echo "🔗 Backend API should be running at: http://localhost:8001"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start dev server
npm run dev
