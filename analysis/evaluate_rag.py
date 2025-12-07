"""
Evaluation script to compare RAG-enabled system vs baseline GPT-4o-mini.
This script tests whether RAG improves answer quality for remote sensing questions.
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import sys

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import get_settings, get_vectorstore_path
from src.vector_store import VectorStoreManager
from src.rag_graph import RAGGraph

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluates and compares RAG system against baseline non-RAG system.
    """
    
    def __init__(self):
        """Initialize evaluator with both RAG and non-RAG systems."""
        logger.info("Initializing RAG Evaluator...")
        
        # Load settings
        self.settings = get_settings()
        
        # Initialize non-RAG baseline (just GPT-4o-mini)
        self.baseline_llm = ChatOpenAI(
            model=self.settings.llm_model,
            temperature=self.settings.temperature,
            openai_api_key=self.settings.openai_api_key
        )
        
        # Initialize RAG system
        vectorstore_path = get_vectorstore_path(self.settings)
        
        if not vectorstore_path.exists():
            raise FileNotFoundError(
                f"Vector store not found at {vectorstore_path}. "
                "Please run 'python main.py --index' first."
            )
        
        # Load vector store
        self.vector_manager = VectorStoreManager(
            embedding_model=self.settings.embedding_model,
            api_key=self.settings.openai_api_key
        )
        self.vector_manager.load_vector_store(vectorstore_path)
        
        # Initialize RAG graph
        self.rag_system = RAGGraph(
            vector_store_manager=self.vector_manager,
            llm_model=self.settings.llm_model,
            temperature=self.settings.temperature,
            top_k=self.settings.top_k_results,
            api_key=self.settings.openai_api_key
        )
        
        logger.info("Evaluator initialized successfully")
    
    def load_test_questions(self, questions_path: Path) -> List[Dict[str, Any]]:
        """
        Load test questions from JSON file.
        
        Args:
            questions_path: Path to test questions JSON file
            
        Returns:
            List of question dictionaries
        """
        logger.info(f"Loading test questions from {questions_path}")
        
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        logger.info(f"Loaded {len(questions)} test questions")
        return questions
    
    def query_baseline(self, question: str) -> Dict[str, Any]:
        """
        Query baseline system (non-RAG, just GPT-4o-mini).
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Querying baseline for: '{question[:50]}...'")
        
        # Simple prompt without any document context
        prompt_template = ChatPromptTemplate.from_template(
            """You are an expert assistant specializing in remote sensing and image segmentation research.
Answer the following question based on your training knowledge:

Question: {question}

Instructions:
1. Provide a comprehensive answer based on your knowledge.
2. Be technical and accurate.
3. If you're uncertain, indicate that clearly.
4. Use appropriate terminology from computer vision and remote sensing.

Answer:"""
        )
        
        start_time = time.time()
        
        try:
            prompt = prompt_template.format_messages(question=question)
            response = self.baseline_llm.invoke(prompt)
            
            elapsed_time = time.time() - start_time
            
            return {
                "answer": response.content,
                "response_time": elapsed_time,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error in baseline query: {str(e)}")
            return {
                "answer": "Error generating answer",
                "response_time": time.time() - start_time,
                "error": str(e)
            }
    
    def query_rag(self, question: str) -> Dict[str, Any]:
        """
        Query RAG system.
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Querying RAG system for: '{question[:50]}...'")
        
        start_time = time.time()
        
        try:
            result = self.rag_system.query(question)
            elapsed_time = time.time() - start_time
            
            return {
                "answer": result["answer"],
                "response_time": elapsed_time,
                "num_sources": len(result["sources"]),
                "sources": result["sources"],
                "error": result.get("errors", [])
            }
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return {
                "answer": "Error generating answer",
                "response_time": time.time() - start_time,
                "num_sources": 0,
                "sources": [],
                "error": str(e)
            }
    
    def evaluate_all_questions(
        self,
        questions: List[Dict[str, Any]],
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Evaluate all test questions with both systems.
        
        Args:
            questions: List of test questions
            output_path: Path to save results
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Starting evaluation of {len(questions)} questions")
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(questions),
                "llm_model": self.settings.llm_model,
                "embedding_model": self.settings.embedding_model,
                "top_k": self.settings.top_k_results,
                "temperature": self.settings.temperature
            },
            "questions": []
        }
        
        for idx, question_data in enumerate(questions, 1):
            question = question_data["question"]
            logger.info(f"\n{'='*80}")
            logger.info(f"Question {idx}/{len(questions)}: {question}")
            logger.info(f"{'='*80}")
            
            # Query baseline
            baseline_result = self.query_baseline(question)
            logger.info(f"✓ Baseline answered in {baseline_result['response_time']:.2f}s")
            
            # Small delay to avoid rate limiting
            time.sleep(1)
            
            # Query RAG
            rag_result = self.query_rag(question)
            logger.info(f"✓ RAG answered in {rag_result['response_time']:.2f}s")
            
            # Store results
            question_result = {
                "id": question_data["id"],
                "question": question,
                "category": question_data["category"],
                "expected_topics": question_data["expected_topics"],
                "baseline": {
                    "answer": baseline_result["answer"],
                    "response_time": baseline_result["response_time"],
                    "answer_length": len(baseline_result["answer"])
                },
                "rag": {
                    "answer": rag_result["answer"],
                    "response_time": rag_result["response_time"],
                    "answer_length": len(rag_result["answer"]),
                    "num_sources": rag_result.get("num_sources", 0),
                    "sources": [
                        {
                            "source": src["source"],
                            "page": src["page"]
                        }
                        for src in rag_result.get("sources", [])
                    ]
                }
            }
            
            results["questions"].append(question_result)
            
            # Small delay between questions
            time.sleep(1)
        
        # Calculate summary statistics
        baseline_times = [q["baseline"]["response_time"] for q in results["questions"]]
        rag_times = [q["rag"]["response_time"] for q in results["questions"]]
        baseline_lengths = [q["baseline"]["answer_length"] for q in results["questions"]]
        rag_lengths = [q["rag"]["answer_length"] for q in results["questions"]]
        
        results["summary"] = {
            "baseline": {
                "avg_response_time": sum(baseline_times) / len(baseline_times),
                "avg_answer_length": sum(baseline_lengths) / len(baseline_lengths),
                "total_time": sum(baseline_times)
            },
            "rag": {
                "avg_response_time": sum(rag_times) / len(rag_times),
                "avg_answer_length": sum(rag_lengths) / len(rag_lengths),
                "total_time": sum(rag_times),
                "avg_sources_used": sum(q["rag"]["num_sources"] for q in results["questions"]) / len(results["questions"])
            }
        }
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*80}")
        logger.info("Evaluation completed!")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"{'='*80}")
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """
        Print summary of evaluation results.
        
        Args:
            results: Evaluation results dictionary
        """
        summary = results["summary"]
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        print("\nBaseline (Non-RAG):")
        print(f"  Average Response Time: {summary['baseline']['avg_response_time']:.2f}s")
        print(f"  Average Answer Length: {summary['baseline']['avg_answer_length']:.0f} chars")
        print(f"  Total Time: {summary['baseline']['total_time']:.2f}s")
        
        print("\nRAG System:")
        print(f"  Average Response Time: {summary['rag']['avg_response_time']:.2f}s")
        print(f"  Average Answer Length: {summary['rag']['avg_answer_length']:.0f} chars")
        print(f"  Average Sources Used: {summary['rag']['avg_sources_used']:.1f} documents")
        print(f"  Total Time: {summary['rag']['total_time']:.2f}s")
        
        print("\nComparison:")
        time_diff = summary['rag']['avg_response_time'] - summary['baseline']['avg_response_time']
        print(f"  Time Difference: {time_diff:+.2f}s ({time_diff/summary['baseline']['avg_response_time']*100:+.1f}%)")
        
        length_diff = summary['rag']['avg_answer_length'] - summary['baseline']['avg_answer_length']
        print(f"  Length Difference: {length_diff:+.0f} chars ({length_diff/summary['baseline']['avg_answer_length']*100:+.1f}%)")
        
        print("="*80 + "\n")


def main():
    """Main execution function."""
    # Paths
    project_root = Path(__file__).parent.parent
    questions_path = Path(__file__).parent / "test_questions.json"
    output_path = Path(__file__).parent / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        # Initialize evaluator
        evaluator = RAGEvaluator()
        
        # Load questions
        questions = evaluator.load_test_questions(questions_path)
        
        # Run evaluation
        results = evaluator.evaluate_all_questions(questions, output_path)
        
        # Print summary
        evaluator.print_summary(results)
        
        print(f"\nDetailed results saved to: {output_path}")
        print("\nNext step: Run 'python analysis/visualize_results.py' to generate visualizations")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

