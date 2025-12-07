"""
Document processing module for loading and chunking PDF documents.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .utils import get_pdf_files, get_file_info


logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles loading and processing of PDF documents.
    Splits documents into chunks for efficient retrieval.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n"
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
            separator: Primary separator for splitting text
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(
            f"DocumentProcessor initialized with chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )
    
    def load_pdf(self, file_path: Path) -> List[Document]:
        """
        Load a single PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects
        """
        try:
            logger.info(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # Add source information to metadata
            for doc in documents:
                doc.metadata['source'] = file_path.name
                doc.metadata['full_path'] = str(file_path)
            
            logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            return []
    
    def load_directory(self, directory_path: Path) -> List[Document]:
        """
        Load all PDF files from a directory recursively.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of Document objects from all PDFs
        """
        logger.info(f"Loading PDFs from directory: {directory_path}")
        
        pdf_files = get_pdf_files(directory_path)
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        all_documents = []
        for pdf_file in pdf_files:
            docs = self.load_pdf(pdf_file)
            all_documents.extend(docs)
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for processing.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        if not documents:
            logger.warning("No documents to chunk")
            return []
        
        logger.info(f"Chunking {len(documents)} documents...")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for idx, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = idx
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def process_documents(self, directory_path: Path) -> List[Document]:
        """
        Complete pipeline: load and chunk documents from a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of chunked Document objects ready for embedding
        """
        logger.info("Starting document processing pipeline...")
        
        # Load documents
        documents = self.load_directory(directory_path)
        
        if not documents:
            logger.error("No documents loaded. Aborting processing.")
            return []
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        logger.info("Document processing pipeline completed")
        return chunks
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about processed documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'total_characters': 0,
                'sources': []
            }
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        sources = set(doc.metadata.get('source', 'Unknown') for doc in documents)
        
        return {
            'total_documents': len(documents),
            'total_chunks': len(documents),
            'avg_chunk_size': total_chars // len(documents) if documents else 0,
            'total_characters': total_chars,
            'sources': sorted(list(sources)),
            'num_sources': len(sources)
        }
    
    def filter_documents_by_source(
        self,
        documents: List[Document],
        source_name: str
    ) -> List[Document]:
        """
        Filter documents by source file name.
        
        Args:
            documents: List of Document objects
            source_name: Name of source file to filter by
            
        Returns:
            Filtered list of Document objects
        """
        filtered = [
            doc for doc in documents
            if doc.metadata.get('source', '').lower() == source_name.lower()
        ]
        
        logger.info(
            f"Filtered {len(filtered)} documents from source '{source_name}'"
        )
        return filtered
    
    def preview_chunk(self, document: Document, max_chars: int = 200) -> str:
        """
        Get a preview of a document chunk.
        
        Args:
            document: Document object
            max_chars: Maximum characters to include in preview
            
        Returns:
            Preview string
        """
        content = document.page_content[:max_chars]
        if len(document.page_content) > max_chars:
            content += "..."
        
        metadata = document.metadata
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page', 'N/A')
        
        return f"[{source} - Page {page}] {content}"

