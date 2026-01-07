#!/bin/bash
# ============================================================
# Data Copilot Lab - Development Run Script
# Starts both API and Frontend in development mode
# ============================================================

set -e

echo "🚀 Starting Data Copilot Lab in development mode..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./scripts/setup.sh first"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys!"
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start API server in background
echo "🔧 Starting FastAPI server..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait a bit for API to start
sleep 3

# Start Streamlit frontend
echo "🎨 Starting Streamlit frontend..."
streamlit run frontend/streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0 &
STREAMLIT_PID=$!

echo ""
echo "✅ Data Copilot Lab is running!"
echo ""
echo "📍 Access points:"
echo "   - Frontend (Streamlit): http://localhost:8501"
echo "   - API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for all background processes
wait
