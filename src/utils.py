"""
Utility functions for the RAG Q&A system.
"""

import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import logging


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def get_pdf_files(directory: Path) -> List[Path]:
    """
    Recursively get all PDF files from a directory.
    
    Args:
        directory: Directory path to search for PDFs
        
    Returns:
        List of PDF file paths
    """
    pdf_files = []
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Recursively find all PDF files
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(Path(root) / file)
    
    return sorted(pdf_files)


def format_documents_for_display(documents: List[dict]) -> str:
    """
    Format retrieved documents for display.
    
    Args:
        documents: List of document dictionaries with content and metadata
        
    Returns:
        Formatted string representation
    """
    formatted = []
    
    for idx, doc in enumerate(documents, 1):
        source = doc.get('source', 'Unknown')
        content = doc.get('content', '')
        page = doc.get('page', 'N/A')
        
        formatted.append(f"\n{'='*80}")
        formatted.append(f"Document {idx}:")
        formatted.append(f"Source: {source}")
        formatted.append(f"Page: {page}")
        formatted.append(f"{'-'*80}")
        formatted.append(content)
        formatted.append(f"{'='*80}\n")
    
    return '\n'.join(formatted)


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """
    Estimate the number of tokens in a text string.
    
    Args:
        text: Input text
        chars_per_token: Average characters per token (default: 4)
        
    Returns:
        Estimated token count
    """
    return len(text) // chars_per_token


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length in characters
        
    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def get_file_info(file_path: Path) -> dict:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    if not file_path.exists():
        return {}
    
    stat = file_path.stat()
    
    return {
        'name': file_path.name,
        'path': str(file_path),
        'size': stat.st_size,
        'size_mb': round(stat.st_size / (1024 * 1024), 2),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
    }


def validate_api_key(api_key: Optional[str]) -> bool:
    """
    Validate OpenAI API key format.
    
    Args:
        api_key: OpenAI API key
        
    Returns:
        True if valid format, False otherwise
    """
    if not api_key:
        return False
    
    # Basic validation: should start with 'sk-' and have reasonable length
    return api_key.startswith('sk-') and len(api_key) > 20


def create_safe_filename(text: str, max_length: int = 100) -> str:
    """
    Create a safe filename from text.
    
    Args:
        text: Input text
        max_length: Maximum filename length
        
    Returns:
        Safe filename string
    """
    # Remove or replace unsafe characters
    safe_chars = []
    for char in text[:max_length]:
        if char.isalnum() or char in (' ', '-', '_', '.'):
            safe_chars.append(char)
        else:
            safe_chars.append('_')
    
    return ''.join(safe_chars).strip()


def print_header(text: str, width: int = 80, char: str = '='):
    """
    Print a formatted header.
    
    Args:
        text: Header text
        width: Width of header
        char: Character to use for border
    """
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")


def print_section(title: str, content: str, width: int = 80):
    """
    Print a formatted section.
    
    Args:
        title: Section title
        content: Section content
        width: Width of section
    """
    print(f"\n{title}")
    print(f"{'-' * width}")
    print(content)
    print(f"{'-' * width}\n")

