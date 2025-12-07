@echo off
REM Start script for Remote Sensing Papers RAG Q&A Web Interface (Windows)

echo ================================================
echo   Remote Sensing Papers RAG Q^&A System
echo   Starting Web Interface...
echo ================================================
echo.

REM Check if vector store exists
if not exist "data\vectorstore\index.faiss" (
    echo [31mError: Vector store not found![0m
    echo.
    echo Please run indexing first:
    echo   python main.py --index
    echo.
    pause
    exit /b 1
)

echo [32m√ Vector store found[0m
echo.
echo Starting Streamlit application...
echo The web interface will open in your browser automatically.
echo.
echo [33mPress Ctrl+C to stop the server[0m
echo.

REM Start Streamlit
streamlit run streamlit_app.py

pause

