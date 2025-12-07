"""
Main CLI application for Remote Sensing Papers RAG Q&A System.
Provides command-line interface for indexing documents and querying the system.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import project modules
from config import get_settings, ensure_directories, get_papers_path, get_vectorstore_path
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_graph import RAGGraph
from src.utils import setup_logging, print_header, print_section


# Initialize rich console
console = Console()


def setup_environment():
    """
    Set up the environment and validate configuration.
    """
    try:
        # Ensure necessary directories exist
        ensure_directories()
        
        # Get settings
        settings = get_settings()
        
        # Validate API key
        if not settings.openai_api_key or settings.openai_api_key == "your-openai-api-key-here":
            console.print(
                "[red]Error: OpenAI API key not configured.[/red]\n"
                "Please set your API key in the .env file.\n"
                "Example: OPENAI_API_KEY=sk-your-actual-key-here"
            )
            sys.exit(1)
        
        return settings
        
    except Exception as e:
        console.print(f"[red]Error during setup: {str(e)}[/red]")
        sys.exit(1)


def index_documents(args):
    """
    Index documents from the papers directory.
    
    Args:
        args: Command-line arguments
    """
    console.print(Panel.fit(
        "[bold cyan]Document Indexing Pipeline[/bold cyan]",
        border_style="cyan"
    ))
    
    settings = setup_environment()
    papers_path = Path(args.papers_dir) if args.papers_dir else get_papers_path(settings)
    vectorstore_path = get_vectorstore_path(settings)
    
    # Override settings if provided
    chunk_size = args.chunk_size if args.chunk_size else settings.chunk_size
    chunk_overlap = args.chunk_overlap if args.chunk_overlap else settings.chunk_overlap
    
    console.print(f"\n[cyan]Papers directory:[/cyan] {papers_path}")
    console.print(f"[cyan]Vector store path:[/cyan] {vectorstore_path}")
    console.print(f"[cyan]Chunk size:[/cyan] {chunk_size}")
    console.print(f"[cyan]Chunk overlap:[/cyan] {chunk_overlap}\n")
    
    # Check if papers directory exists
    if not papers_path.exists():
        console.print(f"[red]Error: Papers directory not found at {papers_path}[/red]")
        sys.exit(1)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Step 1: Load and process documents
            task1 = progress.add_task("[cyan]Loading PDF documents...", total=None)
            doc_processor = DocumentProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = doc_processor.process_documents(papers_path)
            progress.update(task1, completed=True)
            
            if not chunks:
                console.print("[red]No documents were loaded. Please check your papers directory.[/red]")
                sys.exit(1)
            
            # Display statistics
            stats = doc_processor.get_document_stats(chunks)
            console.print(f"\n[green][OK] Loaded {stats['num_sources']} PDF files[/green]")
            console.print(f"[green][OK] Created {stats['total_chunks']} chunks[/green]")
            console.print(f"[green][OK] Average chunk size: {stats['avg_chunk_size']} characters[/green]\n")
            
            # Step 2: Create embeddings and vector store
            task2 = progress.add_task("[cyan]Creating embeddings and vector store...", total=None)
            vector_manager = VectorStoreManager(
                embedding_model=settings.embedding_model,
                api_key=settings.openai_api_key
            )
            vector_manager.create_vector_store(chunks)
            progress.update(task2, completed=True)
            
            # Step 3: Save vector store
            task3 = progress.add_task("[cyan]Saving vector store to disk...", total=None)
            vector_manager.save_vector_store(vectorstore_path)
            progress.update(task3, completed=True)
        
        console.print(f"\n[bold green][OK] Indexing completed successfully![/bold green]")
        console.print(f"Vector store saved to: {vectorstore_path}\n")
        
    except Exception as e:
        console.print(f"\n[red]Error during indexing: {str(e)}[/red]")
        logging.error(f"Indexing error: {str(e)}", exc_info=True)
        sys.exit(1)


def query_system(args):
    """
    Query the RAG system with a question.
    
    Args:
        args: Command-line arguments
    """
    settings = setup_environment()
    vectorstore_path = get_vectorstore_path(settings)
    
    # Check if vector store exists
    if not vectorstore_path.exists():
        console.print(
            "[red]Error: Vector store not found.[/red]\n"
            "Please run indexing first: python main.py --index"
        )
        sys.exit(1)
    
    try:
        # Load vector store
        with console.status("[cyan]Loading vector store..."):
            vector_manager = VectorStoreManager(
                embedding_model=settings.embedding_model,
                api_key=settings.openai_api_key
            )
            vector_manager.load_vector_store(vectorstore_path)
        
        # Initialize RAG graph
        top_k = args.top_k if args.top_k else settings.top_k_results
        rag_graph = RAGGraph(
            vector_store_manager=vector_manager,
            llm_model=settings.llm_model,
            temperature=settings.temperature,
            top_k=top_k,
            api_key=settings.openai_api_key
        )
        
        # Get question
        question = args.query
        
        # Display query
        console.print(Panel.fit(
            f"[bold cyan]Question:[/bold cyan]\n{question}",
            border_style="cyan"
        ))
        
        # Process query
        with console.status("[cyan]Processing query..."):
            result = rag_graph.query(question)
        
        # Display answer
        console.print(Panel(
            Markdown(result["answer"]),
            title="[bold green]Answer[/bold green]",
            border_style="green"
        ))
        
        # Display sources
        if result["sources"]:
            console.print("\n[bold cyan]Sources:[/bold cyan]")
            for idx, source in enumerate(result["sources"], 1):
                console.print(f"\n{idx}. [yellow]{source['source']}[/yellow] (Page {source['page']})")
                console.print(f"   [dim]{source['content_preview']}[/dim]")
        
        # Display errors if any
        if result["errors"]:
            console.print("\n[bold red]Errors:[/bold red]")
            for error in result["errors"]:
                console.print(f"  • {error}")
        
    except Exception as e:
        console.print(f"\n[red]Error during query: {str(e)}[/red]")
        logging.error(f"Query error: {str(e)}", exc_info=True)
        sys.exit(1)


def interactive_mode(args):
    """
    Start interactive Q&A session.
    
    Args:
        args: Command-line arguments
    """
    console.print(Panel.fit(
        "[bold cyan]Interactive RAG Q&A Session[/bold cyan]\n"
        "Type your questions or 'quit' to exit.",
        border_style="cyan"
    ))
    
    settings = setup_environment()
    vectorstore_path = get_vectorstore_path(settings)
    
    # Check if vector store exists
    if not vectorstore_path.exists():
        console.print(
            "[red]Error: Vector store not found.[/red]\n"
            "Please run indexing first: python main.py --index"
        )
        sys.exit(1)
    
    try:
        # Load vector store
        with console.status("[cyan]Loading vector store..."):
            vector_manager = VectorStoreManager(
                embedding_model=settings.embedding_model,
                api_key=settings.openai_api_key
            )
            vector_manager.load_vector_store(vectorstore_path)
        
        # Initialize RAG graph
        top_k = args.top_k if args.top_k else settings.top_k_results
        rag_graph = RAGGraph(
            vector_store_manager=vector_manager,
            llm_model=settings.llm_model,
            temperature=settings.temperature,
            top_k=top_k,
            api_key=settings.openai_api_key
        )
        
        console.print("[green][OK] System ready![/green]\n")
        
        # Interactive loop
        while True:
            try:
                # Get question from user
                question = Prompt.ask("\n[bold cyan]Your question[/bold cyan]")
                
                if question.lower() in ['quit', 'exit', 'q']:
                    console.print("\n[yellow]Goodbye![/yellow]")
                    break
                
                if not question.strip():
                    continue
                
                # Process query
                with console.status("[cyan]Processing query..."):
                    result = rag_graph.query(question)
                
                # Display answer
                console.print(Panel(
                    Markdown(result["answer"]),
                    title="[bold green]Answer[/bold green]",
                    border_style="green"
                ))
                
                # Optionally show sources
                if result["sources"] and Confirm.ask("Show sources?", default=False):
                    console.print("\n[bold cyan]Sources:[/bold cyan]")
                    for idx, source in enumerate(result["sources"], 1):
                        console.print(f"\n{idx}. [yellow]{source['source']}[/yellow] (Page {source['page']})")
                        console.print(f"   [dim]{source['content_preview']}[/dim]")
                
            except KeyboardInterrupt:
                console.print("\n\n[yellow]Session interrupted. Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
                continue
        
    except Exception as e:
        console.print(f"\n[red]Error during interactive session: {str(e)}[/red]")
        logging.error(f"Interactive mode error: {str(e)}", exc_info=True)
        sys.exit(1)


def main():
    """
    Main entry point for the CLI application.
    """
    parser = argparse.ArgumentParser(
        description="Remote Sensing Papers RAG Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index documents
  python main.py --index
  
  # Query the system
  python main.py --query "What are evaluation metrics for panoptic segmentation?"
  
  # Interactive mode
  python main.py --interactive
  
  # Custom papers directory
  python main.py --index --papers-dir /path/to/papers
  
  # Adjust retrieval settings
  python main.py --query "Your question" --top-k 5
        """
    )
    
    # Main command groups
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--index",
        action="store_true",
        help="Index documents from papers directory"
    )
    group.add_argument(
        "--query",
        type=str,
        help="Ask a question to the RAG system"
    )
    group.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive Q&A session"
    )
    
    # Indexing options
    parser.add_argument(
        "--papers-dir",
        type=str,
        help="Path to papers directory (default: from config)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Chunk size for text splitting (default: from config)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        help="Chunk overlap for text splitting (default: from config)"
    )
    
    # Query options
    parser.add_argument(
        "--top-k",
        type=int,
        help="Number of documents to retrieve (default: from config)"
    )
    
    # General options
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Execute appropriate command
    if args.index:
        index_documents(args)
    elif args.query:
        query_system(args)
    elif args.interactive:
        interactive_mode(args)


if __name__ == "__main__":
    main()

