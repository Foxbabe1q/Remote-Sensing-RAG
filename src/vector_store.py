"""
Vector store management module for storing and retrieving document embeddings.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages vector store operations including creation, saving, loading, and retrieval.
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_model: Name of the OpenAI embedding model to use
            api_key: OpenAI API key
        """
        self.embedding_model_name = embedding_model
        self.api_key = api_key
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=api_key
        )
        
        self.vector_store: Optional[FAISS] = None
        
        logger.info(f"VectorStoreManager initialized with model: {embedding_model}")
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of Document objects to embed
            
        Returns:
            FAISS vector store instance
        """
        if not documents:
            raise ValueError("Cannot create vector store from empty document list")
        
        logger.info(f"Creating vector store from {len(documents)} documents...")
        
        try:
            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            logger.info("Vector store created successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def save_vector_store(self, save_path: Path) -> None:
        """
        Save the vector store to disk.
        
        Args:
            save_path: Directory path to save the vector store
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save. Create one first.")
        
        logger.info(f"Saving vector store to {save_path}")
        
        try:
            # Create directory if it doesn't exist
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            self.vector_store.save_local(str(save_path))
            
            logger.info("Vector store saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_vector_store(self, load_path: Path) -> FAISS:
        """
        Load a vector store from disk.
        
        Args:
            load_path: Directory path to load the vector store from
            
        Returns:
            Loaded FAISS vector store
        """
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found at {load_path}")
        
        logger.info(f"Loading vector store from {load_path}")
        
        try:
            # Load FAISS index
            try:
                # Try with allow_dangerous_deserialization parameter (newer versions)
                self.vector_store = FAISS.load_local(
                    str(load_path),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except TypeError:
                # Fall back to without parameter (older versions)
                self.vector_store = FAISS.load_local(
                    str(load_path),
                    embeddings=self.embeddings
                )
            
            logger.info("Vector store loaded successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search to find relevant documents.
        
        Args:
            query: Query string to search for
            k: Number of documents to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of most relevant Document objects
        """
        if self.vector_store is None:
            raise ValueError("No vector store loaded. Load or create one first.")
        
        logger.info(f"Performing similarity search for query: '{query[:50]}...'")
        
        try:
            # Perform similarity search
            if filter_dict:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k
                )
            
            logger.info(f"Found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search and return documents with relevance scores.
        
        Args:
            query: Query string to search for
            k: Number of documents to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of tuples (Document, score)
        """
        if self.vector_store is None:
            raise ValueError("No vector store loaded. Load or create one first.")
        
        logger.info(f"Performing similarity search with scores for: '{query[:50]}...'")
        
        try:
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            logger.info(f"Found {len(results)} relevant documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to existing vector store.
        
        Args:
            documents: List of Document objects to add
        """
        if self.vector_store is None:
            raise ValueError("No vector store exists. Create one first.")
        
        if not documents:
            logger.warning("No documents to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        try:
            self.vector_store.add_documents(documents)
            logger.info("Documents added successfully")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def get_retriever(self, k: int = 3, search_type: str = "similarity"):
        """
        Get a retriever object for use in chains.
        
        Args:
            k: Number of documents to retrieve
            search_type: Type of search ("similarity" or "mmr")
            
        Returns:
            Retriever object
        """
        if self.vector_store is None:
            raise ValueError("No vector store loaded. Load or create one first.")
        
        logger.info(f"Creating retriever with k={k}, search_type={search_type}")
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
    
    def delete_vector_store(self, path: Path) -> None:
        """
        Delete vector store files from disk.
        
        Args:
            path: Path to vector store directory
        """
        logger.info(f"Deleting vector store at {path}")
        
        try:
            if path.exists() and path.is_dir():
                import shutil
                shutil.rmtree(path)
                logger.info("Vector store deleted successfully")
            else:
                logger.warning(f"Vector store not found at {path}")
                
        except Exception as e:
            logger.error(f"Error deleting vector store: {str(e)}")
            raise
    
    def get_store_info(self) -> Dict[str, Any]:
        """
        Get information about the current vector store.
        
        Returns:
            Dictionary with vector store statistics
        """
        if self.vector_store is None:
            return {
                'exists': False,
                'num_vectors': 0,
                'embedding_model': self.embedding_model_name
            }
        
        try:
            # Get number of vectors in store
            num_vectors = self.vector_store.index.ntotal
            
            return {
                'exists': True,
                'num_vectors': num_vectors,
                'embedding_model': self.embedding_model_name,
                'dimension': self.vector_store.index.d if num_vectors > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting store info: {str(e)}")
            return {
                'exists': True,
                'num_vectors': -1,
                'embedding_model': self.embedding_model_name,
                'error': str(e)
            }

