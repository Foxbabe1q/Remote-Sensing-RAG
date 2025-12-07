@echo off
REM Setup script for Remote Sensing Papers RAG Q&A System (Windows)

echo ==================================================
echo   Remote Sensing Papers RAG Q^&A System Setup
echo ==================================================
echo.

REM Check Python version
echo Checking Python version...
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
echo [32m√ Virtual environment created[0m
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo [32m√ Virtual environment activated[0m
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo [32m√ pip upgraded[0m
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
echo [32m√ Dependencies installed[0m
echo.

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file...
    (
        echo # OpenAI API Configuration
        echo OPENAI_API_KEY=your-openai-api-key-here
        echo.
        echo # Model Configuration
        echo EMBEDDING_MODEL=text-embedding-3-small
        echo LLM_MODEL=gpt-4o-mini
        echo TEMPERATURE=0.7
        echo.
        echo # Document Processing
        echo CHUNK_SIZE=1000
        echo CHUNK_OVERLAP=200
        echo TOP_K_RESULTS=3
        echo.
        echo # Paths
        echo PAPERS_DIR=papers
        echo VECTORSTORE_PATH=data/vectorstore
        echo.
        echo # LangChain Configuration
        echo LANGCHAIN_TRACING_V2=false
        echo LANGCHAIN_API_KEY=
    ) > .env
    echo [32m√ .env file created[0m
    echo.
    echo [33m⚠ IMPORTANT: Please edit .env and add your OpenAI API key[0m
) else (
    echo [32m√ .env file already exists[0m
)
echo.

REM Create necessary directories
echo Creating directories...
if not exist data mkdir data
if not exist data\vectorstore mkdir data\vectorstore
echo [32m√ Directories created[0m
echo.

echo ==================================================
echo   Setup completed successfully!
echo ==================================================
echo.
echo Next steps:
echo 1. Edit .env and add your OpenAI API key
echo 2. Place your PDF papers in the 'papers/' directory
echo 3. Run: python main.py --index
echo 4. Query: python main.py --query "Your question"
echo.
pause

