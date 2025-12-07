"""
Example usage of the RAG Q&A System.

This script demonstrates various ways to use the system programmatically.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_settings, get_papers_path, get_vectorstore_path
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_graph import RAGGraph


def example_1_basic_query():
    """
    Example 1: Basic query using existing vector store.
    """
    print("=" * 80)
    print("Example 1: Basic Query")
    print("=" * 80)
    
    # Load settings
    settings = get_settings()
    
    # Initialize vector store manager
    vector_manager = VectorStoreManager(
        embedding_model=settings.embedding_model,
        api_key=settings.openai_api_key
    )
    
    # Load existing vector store
    vectorstore_path = get_vectorstore_path(settings)
    vector_manager.load_vector_store(vectorstore_path)
    
    # Initialize RAG graph
    rag = RAGGraph(
        vector_store_manager=vector_manager,
        llm_model=settings.llm_model,
        temperature=settings.temperature,
        top_k=3,
        api_key=settings.openai_api_key
    )
    
    # Query the system
    question = "What is panoptic segmentation?"
    print(f"\nQuestion: {question}\n")
    
    result = rag.query(question)
    
    print(f"Answer: {result['answer']}\n")
    print(f"Number of sources: {len(result['sources'])}\n")
    
    for idx, source in enumerate(result['sources'], 1):
        print(f"Source {idx}: {source['source']} (Page {source['page']})")


def example_2_multiple_queries():
    """
    Example 2: Ask multiple related questions.
    """
    print("\n" + "=" * 80)
    print("Example 2: Multiple Queries")
    print("=" * 80)
    
    settings = get_settings()
    
    # Initialize components
    vector_manager = VectorStoreManager(
        embedding_model=settings.embedding_model,
        api_key=settings.openai_api_key
    )
    vector_manager.load_vector_store(get_vectorstore_path(settings))
    
    rag = RAGGraph(
        vector_store_manager=vector_manager,
        llm_model=settings.llm_model,
        api_key=settings.openai_api_key
    )
    
    # List of questions
    questions = [
        "What datasets are used for remote sensing segmentation?",
        "How is the Panoptic Quality metric calculated?",
        "What is Mask2Former?"
    ]
    
    # Query each question
    for idx, question in enumerate(questions, 1):
        print(f"\n{idx}. Question: {question}")
        result = rag.query(question)
        print(f"   Answer: {result['answer'][:200]}...\n")


def example_3_batch_processing():
    """
    Example 3: Batch process multiple questions.
    """
    print("\n" + "=" * 80)
    print("Example 3: Batch Processing")
    print("=" * 80)
    
    settings = get_settings()
    
    # Initialize components
    vector_manager = VectorStoreManager(
        embedding_model=settings.embedding_model,
        api_key=settings.openai_api_key
    )
    vector_manager.load_vector_store(get_vectorstore_path(settings))
    
    rag = RAGGraph(
        vector_store_manager=vector_manager,
        llm_model=settings.llm_model,
        api_key=settings.openai_api_key
    )
    
    # Batch questions
    questions = [
        "What is semantic segmentation?",
        "What is instance segmentation?",
        "What is panoptic segmentation?"
    ]
    
    print(f"\nProcessing {len(questions)} questions in batch...\n")
    
    # Batch query
    results = rag.batch_query(questions)
    
    # Display results
    for result in results:
        print(f"Q: {result['question']}")
        print(f"A: {result['answer'][:150]}...")
        print()


def example_4_custom_retrieval():
    """
    Example 4: Custom retrieval settings.
    """
    print("\n" + "=" * 80)
    print("Example 4: Custom Retrieval")
    print("=" * 80)
    
    settings = get_settings()
    
    # Initialize vector store
    vector_manager = VectorStoreManager(
        embedding_model=settings.embedding_model,
        api_key=settings.openai_api_key
    )
    vector_manager.load_vector_store(get_vectorstore_path(settings))
    
    # Query with different top-k values
    question = "What evaluation metrics are used in panoptic segmentation?"
    
    for top_k in [2, 3, 5]:
        print(f"\n--- Retrieving top {top_k} documents ---")
        
        rag = RAGGraph(
            vector_store_manager=vector_manager,
            llm_model=settings.llm_model,
            top_k=top_k,
            api_key=settings.openai_api_key
        )
        
        result = rag.query(question)
        print(f"Answer length: {len(result['answer'])} characters")
        print(f"Sources used: {len(result['sources'])}")


def example_5_similarity_search():
    """
    Example 5: Direct similarity search without answer generation.
    """
    print("\n" + "=" * 80)
    print("Example 5: Direct Similarity Search")
    print("=" * 80)
    
    settings = get_settings()
    
    # Initialize vector store
    vector_manager = VectorStoreManager(
        embedding_model=settings.embedding_model,
        api_key=settings.openai_api_key
    )
    vector_manager.load_vector_store(get_vectorstore_path(settings))
    
    # Perform similarity search
    query = "multimodal fusion in remote sensing"
    documents = vector_manager.similarity_search(query, k=3)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(documents)} relevant documents:\n")
    
    for idx, doc in enumerate(documents, 1):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        content_preview = doc.page_content[:150].replace('\n', ' ')
        
        print(f"{idx}. {source} (Page {page})")
        print(f"   {content_preview}...\n")


def example_6_document_statistics():
    """
    Example 6: Get statistics about indexed documents.
    """
    print("\n" + "=" * 80)
    print("Example 6: Document Statistics")
    print("=" * 80)
    
    settings = get_settings()
    
    # Initialize vector store
    vector_manager = VectorStoreManager(
        embedding_model=settings.embedding_model,
        api_key=settings.openai_api_key
    )
    vector_manager.load_vector_store(get_vectorstore_path(settings))
    
    # Get store information
    info = vector_manager.get_store_info()
    
    print("\nVector Store Statistics:")
    print(f"  - Number of vectors: {info['num_vectors']}")
    print(f"  - Embedding model: {info['embedding_model']}")
    print(f"  - Vector dimension: {info['dimension']}")
    print(f"  - Store exists: {info['exists']}")


def example_7_custom_temperature():
    """
    Example 7: Experiment with different temperature settings.
    """
    print("\n" + "=" * 80)
    print("Example 7: Custom Temperature")
    print("=" * 80)
    
    settings = get_settings()
    
    # Initialize vector store
    vector_manager = VectorStoreManager(
        embedding_model=settings.embedding_model,
        api_key=settings.openai_api_key
    )
    vector_manager.load_vector_store(get_vectorstore_path(settings))
    
    question = "What is the main challenge in remote sensing segmentation?"
    
    # Try different temperatures
    temperatures = [0.0, 0.5, 1.0]
    
    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        
        rag = RAGGraph(
            vector_store_manager=vector_manager,
            llm_model=settings.llm_model,
            temperature=temp,
            top_k=3,
            api_key=settings.openai_api_key
        )
        
        result = rag.query(question)
        print(f"Answer: {result['answer'][:200]}...\n")


def example_8_error_handling():
    """
    Example 8: Demonstrate error handling.
    """
    print("\n" + "=" * 80)
    print("Example 8: Error Handling")
    print("=" * 80)
    
    settings = get_settings()
    
    try:
        # Initialize components
        vector_manager = VectorStoreManager(
            embedding_model=settings.embedding_model,
            api_key=settings.openai_api_key
        )
        vector_manager.load_vector_store(get_vectorstore_path(settings))
        
        rag = RAGGraph(
            vector_store_manager=vector_manager,
            llm_model=settings.llm_model,
            api_key=settings.openai_api_key
        )
        
        # Query
        result = rag.query("What is panoptic segmentation?")
        
        # Check for errors
        if result['errors']:
            print("\nErrors encountered:")
            for error in result['errors']:
                print(f"  - {error}")
        else:
            print("\nQuery successful!")
            print(f"Answer: {result['answer'][:100]}...")
            
    except FileNotFoundError:
        print("\nError: Vector store not found. Please run indexing first.")
    except Exception as e:
        print(f"\nError: {str(e)}")


def main():
    """
    Run all examples.
    """
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " RAG Q&A System - Usage Examples ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    try:
        # Run examples
        example_1_basic_query()
        example_2_multiple_queries()
        example_3_batch_processing()
        example_4_custom_retrieval()
        example_5_similarity_search()
        example_6_document_statistics()
        example_7_custom_temperature()
        example_8_error_handling()
        
        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80 + "\n")
        
    except FileNotFoundError:
        print("\n" + "=" * 80)
        print("ERROR: Vector store not found!")
        print("=" * 80)
        print("\nPlease run indexing first:")
        print("  python main.py --index")
        print()
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

