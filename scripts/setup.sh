#!/bin/bash
# ============================================================
# Data Copilot Lab - Setup Script
# Initializes the development environment
# ============================================================

set -e

echo "🚀 Setting up Data Copilot Lab..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔐 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys!"
else
    echo "✓ .env file already exists"
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/raw data/processed data/models data/reports logs

# Initialize database (placeholder)
echo "🗄️  Database initialization skipped (will be done in init_db.py)"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Edit .env file with your configuration"
echo "3. Run the application: ./scripts/run_dev.sh"
echo ""
