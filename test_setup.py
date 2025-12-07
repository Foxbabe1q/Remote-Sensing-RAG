"""
Test script to verify the RAG Q&A system setup.
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    required_packages = [
        ('langchain', 'LangChain'),
        ('langchain_openai', 'LangChain OpenAI'),
        ('langchain_community', 'LangChain Community'),
        ('langgraph', 'LangGraph'),
        ('faiss', 'FAISS'),
        ('pypdf', 'PyPDF'),
        ('openai', 'OpenAI'),
        ('dotenv', 'python-dotenv'),
        ('rich', 'Rich'),
        ('pydantic', 'Pydantic'),
    ]
    
    failed = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  [OK] {name}")
        except ImportError as e:
            print(f"  [FAIL] {name}: {str(e)}")
            failed.append(name)
    
    return len(failed) == 0, failed


def test_config():
    """Test if configuration loads correctly."""
    print("\nTesting configuration...")
    
    try:
        from config import get_settings, ensure_directories
        
        # Try to load settings
        settings = get_settings()
        print(f"  [OK] Configuration loaded")
        print(f"    - Embedding model: {settings.embedding_model}")
        print(f"    - LLM model: {settings.llm_model}")
        print(f"    - Chunk size: {settings.chunk_size}")
        
        # Check API key
        if not settings.openai_api_key or settings.openai_api_key == "your-openai-api-key-here":
            print("  [WARN] OpenAI API key not configured (this is expected for initial setup)")
            return False, "API key not configured"
        else:
            print("  [OK] OpenAI API key configured")
        
        # Ensure directories
        ensure_directories()
        print("  [OK] Directories created/verified")
        
        return True, None
        
    except Exception as e:
        print(f"  [FAIL] Configuration error: {str(e)}")
        return False, str(e)


def test_modules():
    """Test if project modules can be imported."""
    print("\nTesting project modules...")
    
    modules = [
        ('src.document_processor', 'DocumentProcessor'),
        ('src.vector_store', 'VectorStoreManager'),
        ('src.rag_graph', 'RAGGraph'),
        ('src.utils', 'Utilities'),
    ]
    
    failed = []
    for module, name in modules:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError as e:
            print(f"  [FAIL] {name}: {str(e)}")
            failed.append(name)
    
    return len(failed) == 0, failed


def test_directories():
    """Test if required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        ('src', 'Source code directory'),
        ('papers', 'Papers directory'),
        ('data', 'Data directory'),
    ]
    
    missing = []
    for dir_path, description in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  [OK] {description}: {dir_path}")
        else:
            print(f"  [FAIL] {description}: {dir_path} (missing)")
            missing.append(dir_path)
    
    return len(missing) == 0, missing


def test_pdf_files():
    """Test if PDF files exist in papers directory."""
    print("\nTesting PDF files...")
    
    papers_dir = Path("papers")
    if not papers_dir.exists():
        print("  [FAIL] Papers directory not found")
        return False, "Papers directory not found"
    
    # Find all PDF files recursively
    pdf_files = list(papers_dir.rglob("*.pdf"))
    
    if not pdf_files:
        print("  [WARN] No PDF files found in papers directory")
        return False, "No PDFs found"
    
    print(f"  [OK] Found {len(pdf_files)} PDF file(s)")
    
    # Show first few files
    for pdf in pdf_files[:5]:
        print(f"    - {pdf.name}")
    
    if len(pdf_files) > 5:
        print(f"    ... and {len(pdf_files) - 5} more")
    
    return True, None


def main():
    """Run all tests."""
    print("="*60)
    print("  RAG Q&A System Setup Verification")
    print("="*60)
    
    results = []
    
    # Test imports
    success, info = test_imports()
    results.append(("Package imports", success, info))
    
    # Test configuration
    success, info = test_config()
    results.append(("Configuration", success, info))
    
    # Test modules
    success, info = test_modules()
    results.append(("Project modules", success, info))
    
    # Test directories
    success, info = test_directories()
    results.append(("Directory structure", success, info))
    
    # Test PDF files
    success, info = test_pdf_files()
    results.append(("PDF files", success, info))
    
    # Summary
    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    
    all_passed = True
    for test_name, success, info in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status:10} {test_name}")
        if not success and info:
            print(f"           {info}")
        if not success:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n[OK] All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run: python main.py --index")
        print("2. Query: python main.py --query 'Your question'")
        print("3. Interactive: python main.py --interactive")
        return 0
    else:
        print("\n[WARN] Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure API key in .env file")
        print("3. Add PDF files to papers/ directory")
        return 1


if __name__ == "__main__":
    sys.exit(main())

