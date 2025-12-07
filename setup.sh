#!/bin/bash
# Setup script for Remote Sensing Papers RAG Q&A System

echo "=================================================="
echo "  Remote Sensing Papers RAG Q&A System Setup"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
TEMPERATURE=0.7

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=3

# Paths
PAPERS_DIR=papers
VECTORSTORE_PATH=data/vectorstore

# LangChain Configuration
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
EOF
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env and add your OpenAI API key"
else
    echo "✓ .env file already exists"
fi
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p data/vectorstore
echo "✓ Directories created"
echo ""

echo "=================================================="
echo "  Setup completed successfully!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OpenAI API key"
echo "2. Place your PDF papers in the 'papers/' directory"
echo "3. Run: python main.py --index"
echo "4. Query: python main.py --query 'Your question'"
echo ""

