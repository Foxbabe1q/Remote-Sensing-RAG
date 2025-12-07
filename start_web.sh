#!/bin/bash
# Start script for Remote Sensing Papers RAG Q&A Web Interface (Linux/Mac)

echo "================================================"
echo "  Remote Sensing Papers RAG Q&A System"
echo "  Starting Web Interface..."
echo "================================================"
echo ""

# Check if vector store exists
if [ ! -f "data/vectorstore/index.faiss" ]; then
    echo -e "\033[31mError: Vector store not found!\033[0m"
    echo ""
    echo "Please run indexing first:"
    echo "  python main.py --index"
    echo ""
    exit 1
fi

echo -e "\033[32m√ Vector store found\033[0m"
echo ""
echo "Starting Streamlit application..."
echo "The web interface will open in your browser automatically."
echo ""
echo -e "\033[33mPress Ctrl+C to stop the server\033[0m"
echo ""

# Start Streamlit
streamlit run streamlit_app.py

