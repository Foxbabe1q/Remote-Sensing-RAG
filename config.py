"""
Configuration management for RAG Q&A System.
Loads environment variables and provides configuration settings.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI API Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # Model Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small",
        env="EMBEDDING_MODEL"
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        env="LLM_MODEL"
    )
    temperature: float = Field(
        default=0.7,
        env="TEMPERATURE"
    )
    
    # Document Processing Configuration
    chunk_size: int = Field(
        default=1000,
        env="CHUNK_SIZE"
    )
    chunk_overlap: int = Field(
        default=200,
        env="CHUNK_OVERLAP"
    )
    top_k_results: int = Field(
        default=3,
        env="TOP_K_RESULTS"
    )
    
    # Path Configuration
    papers_dir: str = Field(
        default="papers",
        env="PAPERS_DIR"
    )
    vectorstore_path: str = Field(
        default="data/vectorstore",
        env="VECTORSTORE_PATH"
    )
    
    # LangChain Configuration (optional)
    langchain_tracing_v2: bool = Field(
        default=False,
        env="LANGCHAIN_TRACING_V2"
    )
    langchain_api_key: Optional[str] = Field(
        default=None,
        env="LANGCHAIN_API_KEY"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """
    Get application settings instance.
    
    Returns:
        Settings: Configuration settings object
    """
    return Settings()


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path: Project root directory path
    """
    return Path(__file__).parent


def get_papers_path(settings: Optional[Settings] = None) -> Path:
    """
    Get the full path to the papers directory.
    
    Args:
        settings: Optional settings object
        
    Returns:
        Path: Full path to papers directory
    """
    if settings is None:
        settings = get_settings()
    return get_project_root() / settings.papers_dir


def get_vectorstore_path(settings: Optional[Settings] = None) -> Path:
    """
    Get the full path to the vector store directory.
    
    Args:
        settings: Optional settings object
        
    Returns:
        Path: Full path to vector store directory
    """
    if settings is None:
        settings = get_settings()
    return get_project_root() / settings.vectorstore_path


def ensure_directories():
    """Create necessary directories if they don't exist."""
    settings = get_settings()
    
    # Create data directory
    data_dir = get_project_root() / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Create vectorstore directory
    vectorstore_dir = get_vectorstore_path(settings)
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure papers directory exists
    papers_dir = get_papers_path(settings)
    if not papers_dir.exists():
        papers_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created papers directory at: {papers_dir}")

