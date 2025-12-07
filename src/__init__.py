"""
RAG Q&A System for Remote Sensing Papers
A retrieval-augmented generation system built with LangChain and LangGraph.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .document_processor import DocumentProcessor
from .vector_store import VectorStoreManager
from .rag_graph import RAGGraph

__all__ = [
    "DocumentProcessor",
    "VectorStoreManager",
    "RAGGraph",
]

