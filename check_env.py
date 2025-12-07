"""
Quick environment check for Streamlit app.
"""

import os
from pathlib import Path

print("=" * 60)
print("Environment Check for Streamlit Web Interface")
print("=" * 60)
print()

# Check 1: .env file
print("[1/5] Checking .env file...")
env_path = Path(".env")
if env_path.exists():
    print("    [OK] .env file exists")
else:
    print("    [FAIL] .env file not found")
    exit(1)

# Check 2: Load environment variables
print("[2/5] Loading environment variables...")
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if api_key and len(api_key) > 20:
    print(f"    [OK] API key loaded: {api_key[:20]}...")
else:
    print("    [FAIL] API key not found or invalid")
    exit(1)

# Check 3: Import config
print("[3/5] Importing configuration...")
try:
    from config import get_settings
    settings = get_settings()
    print("    [OK] Configuration loaded")
    print(f"    - Embedding: {settings.embedding_model}")
    print(f"    - LLM: {settings.llm_model}")
except Exception as e:
    print(f"    [FAIL] Configuration error: {e}")
    exit(1)

# Check 4: Vector store
print("[4/5] Checking vector store...")
from config import get_vectorstore_path
vectorstore_path = get_vectorstore_path(settings)
if vectorstore_path.exists():
    index_file = vectorstore_path / "index.faiss"
    pkl_file = vectorstore_path / "index.pkl"
    if index_file.exists() and pkl_file.exists():
        print(f"    [OK] Vector store ready at {vectorstore_path}")
    else:
        print(f"    [WARN] Vector store incomplete")
else:
    print(f"    [FAIL] Vector store not found at {vectorstore_path}")
    print("    Please run: python main.py --index")
    exit(1)

# Check 5: Import Streamlit app
print("[5/5] Importing Streamlit app...")
try:
    import streamlit_app
    print("    [OK] Streamlit app imports successfully")
except Exception as e:
    print(f"    [FAIL] Streamlit app error: {e}")
    exit(1)

print()
print("=" * 60)
print("[SUCCESS] All checks passed!")
print("=" * 60)
print()
print("Ready to start the web interface:")
print()
print("  streamlit run streamlit_app.py")
print()
print("or use the startup script:")
print()
print("  start_web.bat   # Windows")
print("  ./start_web.sh  # Linux/Mac")
print()

