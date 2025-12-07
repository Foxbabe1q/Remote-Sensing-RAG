"""
Test script for Streamlit Web Interface.

This script tests the web interface components without actually starting the server.
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("=" * 60)
    print("Testing Imports...")
    print("=" * 60)
    
    try:
        import streamlit as st
        print("[OK] Streamlit imported successfully")
        print(f"    Version: {st.__version__}")
    except ImportError as e:
        print(f"[FAIL] Streamlit import failed: {e}")
        return False
    
    try:
        from config import get_settings, get_vectorstore_path
        print("[OK] Config module imported")
    except ImportError as e:
        print(f"[FAIL] Config import failed: {e}")
        return False
    
    try:
        from src.vector_store import VectorStoreManager
        print("[OK] VectorStoreManager imported")
    except ImportError as e:
        print(f"[FAIL] VectorStoreManager import failed: {e}")
        return False
    
    try:
        from src.rag_graph import RAGGraph
        print("[OK] RAGGraph imported")
    except ImportError as e:
        print(f"[FAIL] RAGGraph import failed: {e}")
        return False
    
    print()
    return True


def test_streamlit_app():
    """Test if streamlit_app.py can be imported."""
    print("=" * 60)
    print("Testing Streamlit App...")
    print("=" * 60)
    
    try:
        import streamlit_app
        print("[OK] streamlit_app.py imported successfully")
        
        # Check if main functions exist
        functions = [
            'initialize_system',
            'render_header',
            'render_sidebar',
            'render_home_page',
            'render_qa_interface',
            'render_stats_page',
            'render_about_page',
            'main'
        ]
        
        for func_name in functions:
            if hasattr(streamlit_app, func_name):
                print(f"[OK] Function '{func_name}' exists")
            else:
                print(f"[WARN] Function '{func_name}' not found")
        
        print()
        return True
        
    except Exception as e:
        print(f"[FAIL] Error importing streamlit_app: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_system_initialization():
    """Test if the system can be initialized."""
    print("=" * 60)
    print("Testing System Initialization...")
    print("=" * 60)
    
    try:
        from config import get_settings, get_vectorstore_path
        
        settings = get_settings()
        print("[OK] Settings loaded successfully")
        
        vectorstore_path = get_vectorstore_path(settings)
        print(f"[OK] Vector store path: {vectorstore_path}")
        
        if vectorstore_path.exists():
            print("[OK] Vector store exists")
            
            from src.vector_store import VectorStoreManager
            
            vector_manager = VectorStoreManager(
                embedding_model=settings.embedding_model,
                api_key=settings.openai_api_key
            )
            print("[OK] VectorStoreManager created")
            
            vector_manager.load_vector_store(vectorstore_path)
            print("[OK] Vector store loaded successfully")
            
            store_info = vector_manager.get_store_info()
            print(f"[OK] Vector store info:")
            print(f"    - Vectors: {store_info['num_vectors']}")
            print(f"    - Dimension: {store_info['dimension']}")
            print(f"    - Model: {store_info['embedding_model']}")
            
            from src.rag_graph import RAGGraph
            
            rag_graph = RAGGraph(
                vector_store_manager=vector_manager,
                llm_model=settings.llm_model,
                temperature=settings.temperature,
                top_k=settings.top_k_results,
                api_key=settings.openai_api_key
            )
            print("[OK] RAG Graph initialized")
            
            print()
            return True
        else:
            print(f"[WARN] Vector store not found at {vectorstore_path}")
            print("       Please run: python main.py --index")
            print()
            return False
            
    except Exception as e:
        print(f"[FAIL] System initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_startup_scripts():
    """Test if startup scripts exist."""
    print("=" * 60)
    print("Testing Startup Scripts...")
    print("=" * 60)
    
    scripts = [
        ('start_web.bat', 'Windows startup script'),
        ('start_web.sh', 'Linux/Mac startup script')
    ]
    
    all_exist = True
    for script_name, description in scripts:
        script_path = Path(script_name)
        if script_path.exists():
            print(f"[OK] {description}: {script_name}")
        else:
            print(f"[FAIL] {description} not found: {script_name}")
            all_exist = False
    
    print()
    return all_exist


def test_documentation():
    """Test if documentation files exist."""
    print("=" * 60)
    print("Testing Documentation...")
    print("=" * 60)
    
    docs = [
        ('WEB_INTERFACE.md', 'Web interface guide'),
        ('WEB_INTERFACE_QUICKSTART.md', 'Web quick start'),
        ('README.md', 'Main readme'),
        ('QUICKSTART.md', 'Quick start guide')
    ]
    
    all_exist = True
    for doc_name, description in docs:
        doc_path = Path(doc_name)
        if doc_path.exists():
            size = doc_path.stat().st_size / 1024
            print(f"[OK] {description}: {doc_name} ({size:.1f} KB)")
        else:
            print(f"[FAIL] {description} not found: {doc_name}")
            all_exist = False
    
    print()
    return all_exist


def test_query_simulation():
    """Simulate a query to test the full pipeline."""
    print("=" * 60)
    print("Testing Query Simulation...")
    print("=" * 60)
    
    try:
        from config import get_settings, get_vectorstore_path
        from src.vector_store import VectorStoreManager
        from src.rag_graph import RAGGraph
        
        settings = get_settings()
        vectorstore_path = get_vectorstore_path(settings)
        
        if not vectorstore_path.exists():
            print("[SKIP] Vector store not found, skipping query test")
            print()
            return True
        
        print("[INFO] Initializing system...")
        
        vector_manager = VectorStoreManager(
            embedding_model=settings.embedding_model,
            api_key=settings.openai_api_key
        )
        vector_manager.load_vector_store(vectorstore_path)
        
        rag_graph = RAGGraph(
            vector_store_manager=vector_manager,
            llm_model=settings.llm_model,
            temperature=0.7,
            top_k=3,
            api_key=settings.openai_api_key
        )
        
        print("[INFO] System initialized, running test query...")
        
        # Simple test query
        test_question = "What is panoptic segmentation?"
        result = rag_graph.query(test_question)
        
        print(f"[OK] Query successful!")
        print(f"    Question: {test_question}")
        print(f"    Answer length: {len(result['answer'])} characters")
        print(f"    Sources found: {len(result['sources'])}")
        
        if result['errors']:
            print(f"[WARN] Errors encountered: {result['errors']}")
        else:
            print(f"[OK] No errors")
        
        print()
        return True
        
    except Exception as e:
        print(f"[FAIL] Query simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results):
    """Print test summary."""
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    failed_tests = total_tests - passed_tests
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status:8} {test_name}")
    
    print("=" * 60)
    print(f"Total: {total_tests} | Passed: {passed_tests} | Failed: {failed_tests}")
    print("=" * 60)
    
    if failed_tests == 0:
        print("\n[OK] All tests passed! Web interface is ready to use.")
        print("\nTo start the web interface:")
        print("  streamlit run streamlit_app.py")
        print("\nor use the startup script:")
        print("  start_web.bat   # Windows")
        print("  ./start_web.sh  # Linux/Mac")
    else:
        print(f"\n[WARN] {failed_tests} test(s) failed. Please fix the issues above.")
    
    return failed_tests == 0


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " Streamlit Web Interface Test Suite ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    results = {}
    
    # Run tests
    results['Imports'] = test_imports()
    results['Streamlit App'] = test_streamlit_app()
    results['System Initialization'] = test_system_initialization()
    results['Startup Scripts'] = test_startup_scripts()
    results['Documentation'] = test_documentation()
    results['Query Simulation'] = test_query_simulation()
    
    # Print summary
    success = print_summary(results)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

