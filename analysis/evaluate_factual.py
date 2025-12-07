"""
Evaluation script for factual precision questions.
Tests RAG vs baseline on questions requiring exact numbers and specific facts.
"""

import json
import time
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import get_settings, get_vectorstore_path
from src.vector_store import VectorStoreManager
from src.rag_graph import RAGGraph

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text for comparison."""
    # Match numbers with optional commas and decimals
    pattern = r'\d+(?:,\d{3})*(?:\.\d+)?'
    matches = re.findall(pattern, text)
    return [float(m.replace(',', '')) for m in matches]


def check_factual_accuracy(answer: str, ground_truth: str, question: str, llm_judge) -> Dict[str, Any]:
    """
    Check if answer contains the key facts from ground truth.
    Uses both automated metrics and LLM-as-a-Judge for comprehensive evaluation.
    
    Args:
        answer: The generated answer to evaluate
        ground_truth: The correct/expected answer
        question: The original question
        llm_judge: LLM instance for judging answer correctness
    
    Returns:
        Dictionary with accuracy metrics and LLM judgment
    """
    # Extract numbers from both
    answer_numbers = set(extract_numbers(answer))
    truth_numbers = set(extract_numbers(ground_truth))
    
    # Calculate number overlap
    if truth_numbers:
        correct_numbers = answer_numbers & truth_numbers
        precision = len(correct_numbers) / len(answer_numbers) if answer_numbers else 0
        recall = len(correct_numbers) / len(truth_numbers) if truth_numbers else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        precision = recall = f1 = 0
    
    # Check for key terms (case-insensitive)
    truth_words = set(ground_truth.lower().split())
    answer_words = set(answer.lower().split())
    term_overlap = len(truth_words & answer_words) / len(truth_words) if truth_words else 0
    
    # LLM-as-a-Judge evaluation
    judge_prompt = ChatPromptTemplate.from_template(
        """You are an expert evaluator for question-answering systems. Your task is to determine if a generated answer is factually correct compared to the ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {generated_answer}

Evaluate whether the generated answer is CORRECT by checking:
1. Are the key numerical values accurate (allow for minor rounding, e.g., "approximately" or "around")?
2. Are the main facts and concepts present?
3. Is the information consistent with the ground truth (even if phrased differently)?

Important: An answer can be marked as CORRECT even if:
- It's more detailed or verbose than the ground truth
- It uses slightly different wording
- Numbers are preceded by "approximately", "around", "roughly" when the exact number is provided
- It includes additional relevant context from the source paper

Respond in the following JSON format:
{{
  "is_correct": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of your judgment",
  "missing_facts": ["list", "of", "missing", "key", "facts"] or [],
  "incorrect_facts": ["list", "of", "incorrect", "facts"] or []
}}

Your evaluation:"""
    )
    
    try:
        judge_messages = judge_prompt.format_messages(
            question=question,
            ground_truth=ground_truth,
            generated_answer=answer
        )
        judge_response = llm_judge.invoke(judge_messages)
        
        # Parse JSON response
        import json
        judge_result = json.loads(judge_response.content)
        llm_judgment = {
            'is_correct': judge_result.get('is_correct', False),
            'confidence': judge_result.get('confidence', 0.0),
            'reasoning': judge_result.get('reasoning', ''),
            'missing_facts': judge_result.get('missing_facts', []),
            'incorrect_facts': judge_result.get('incorrect_facts', [])
        }
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        # Fallback to simple heuristic
        llm_judgment = {
            'is_correct': recall > 0.7 and term_overlap > 0.5,
            'confidence': 0.5,
            'reasoning': 'Fallback evaluation (LLM judge failed)',
            'missing_facts': [],
            'incorrect_facts': []
        }
    
    return {
        'numbers_in_truth': len(truth_numbers),
        'numbers_in_answer': len(answer_numbers),
        'correct_numbers': len(correct_numbers) if truth_numbers else 0,
        'number_precision': precision,
        'number_recall': recall,
        'number_f1': f1,
        'term_overlap': term_overlap,
        'contains_all_numbers': truth_numbers.issubset(answer_numbers) if truth_numbers else True,
        'llm_judgment': llm_judgment
    }


class FactualEvaluator:
    """Evaluates factual precision of RAG vs baseline."""
    
    def __init__(self):
        logger.info("Initializing Factual Evaluator...")
        
        self.settings = get_settings()
        
        # Baseline LLM
        self.baseline_llm = ChatOpenAI(
            model=self.settings.llm_model,
            temperature=self.settings.temperature,
            openai_api_key=self.settings.openai_api_key
        )
        
        # LLM Judge for evaluating answer correctness
        # Using same model with temperature 0 for consistent judgments
        self.llm_judge = ChatOpenAI(
            model=self.settings.llm_model,
            temperature=0.0,  # Deterministic for evaluation
            openai_api_key=self.settings.openai_api_key
        )
        
        # RAG system
        vectorstore_path = get_vectorstore_path(self.settings)
        if not vectorstore_path.exists():
            raise FileNotFoundError(f"Vector store not found at {vectorstore_path}")
        
        self.vector_manager = VectorStoreManager(
            embedding_model=self.settings.embedding_model,
            api_key=self.settings.openai_api_key
        )
        self.vector_manager.load_vector_store(vectorstore_path)
        
        self.rag_system = RAGGraph(
            vector_store_manager=self.vector_manager,
            llm_model=self.settings.llm_model,
            temperature=self.settings.temperature,
            top_k=self.settings.top_k_results,
            api_key=self.settings.openai_api_key
        )
        
        logger.info(f"Evaluator initialized successfully")
        logger.info(f"Answer Generator: {self.settings.llm_model} (temp={self.settings.temperature})")
        logger.info(f"LLM Judge: {self.settings.llm_model} (temp=0.0)")
    
    def load_questions(self, questions_path: Path) -> List[Dict[str, Any]]:
        """Load factual questions from JSON."""
        logger.info(f"Loading questions from {questions_path}")
        
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        logger.info(f"Loaded {len(questions)} factual questions")
        return questions
    
    def query_baseline(self, question: str) -> Dict[str, Any]:
        """Query baseline (no RAG)."""
        prompt_template = ChatPromptTemplate.from_template(
            """You are an expert in remote sensing and computer vision.
Answer the following question based on your knowledge. Be as precise as possible with numbers and facts.

Question: {question}

Provide a concise, factual answer with specific numbers if applicable.

Answer:"""
        )
        
        start_time = time.time()
        
        try:
            prompt = prompt_template.format_messages(question=question)
            response = self.baseline_llm.invoke(prompt)
            
            return {
                "answer": response.content,
                "response_time": time.time() - start_time,
                "error": None
            }
        except Exception as e:
            logger.error(f"Error in baseline query: {e}")
            return {
                "answer": "Error",
                "response_time": time.time() - start_time,
                "error": str(e)
            }
    
    def query_rag(self, question: str) -> Dict[str, Any]:
        """Query RAG system."""
        start_time = time.time()
        
        try:
            result = self.rag_system.query(question)
            
            return {
                "answer": result["answer"],
                "response_time": time.time() - start_time,
                "num_sources": len(result["sources"]),
                "sources": result["sources"],
                "error": result.get("errors", [])
            }
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return {
                "answer": "Error",
                "response_time": time.time() - start_time,
                "num_sources": 0,
                "sources": [],
                "error": str(e)
            }
    
    def evaluate(self, questions: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
        """Run full evaluation."""
        logger.info(f"Starting evaluation of {len(questions)} factual questions")
        
        results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_questions": len(questions),
                "llm_model": self.settings.llm_model,
                "embedding_model": self.settings.embedding_model,
                "evaluation_type": "factual_precision"
            },
            "questions": []
        }
        
        for idx, q_data in enumerate(questions, 1):
            question = q_data["question"]
            ground_truth = q_data["ground_truth"]
            
            logger.info(f"\n{'='*80}")
            logger.info(f"Question {idx}/{len(questions)}: {question[:60]}...")
            logger.info(f"{'='*80}")
            
            # Query baseline
            baseline_result = self.query_baseline(question)
            logger.info(f"✓ Baseline answered in {baseline_result['response_time']:.2f}s")
            
            time.sleep(1)
            
            # Query RAG
            rag_result = self.query_rag(question)
            logger.info(f"✓ RAG answered in {rag_result['response_time']:.2f}s")
            
            # Check accuracy using both automated metrics and LLM judge
            baseline_accuracy = check_factual_accuracy(
                baseline_result['answer'], ground_truth, question, self.llm_judge
            )
            time.sleep(0.5)  # Small delay between judge calls
            
            rag_accuracy = check_factual_accuracy(
                rag_result['answer'], ground_truth, question, self.llm_judge
            )
            
            logger.info(f"  Baseline - Number Recall: {baseline_accuracy['number_recall']:.2%}, LLM Judge: {'✓' if baseline_accuracy['llm_judgment']['is_correct'] else '✗'} ({baseline_accuracy['llm_judgment']['confidence']:.0%})")
            logger.info(f"  RAG      - Number Recall: {rag_accuracy['number_recall']:.2%}, LLM Judge: {'✓' if rag_accuracy['llm_judgment']['is_correct'] else '✗'} ({rag_accuracy['llm_judgment']['confidence']:.0%})")
            
            # Store results
            question_result = {
                "id": q_data["id"],
                "question": question,
                "ground_truth": ground_truth,
                "category": q_data["category"],
                "source": q_data["source"],
                "baseline": {
                    "answer": baseline_result["answer"],
                    "response_time": baseline_result["response_time"],
                    "answer_length": len(baseline_result["answer"]),
                    "accuracy": baseline_accuracy
                },
                "rag": {
                    "answer": rag_result["answer"],
                    "response_time": rag_result["response_time"],
                    "answer_length": len(rag_result["answer"]),
                    "num_sources": rag_result.get("num_sources", 0),
                    "sources": rag_result.get("sources", []),
                    "accuracy": rag_accuracy
                }
            }
            
            results["questions"].append(question_result)
            time.sleep(1)
        
        # Calculate summary statistics
        baseline_times = [q["baseline"]["response_time"] for q in results["questions"]]
        rag_times = [q["rag"]["response_time"] for q in results["questions"]]
        
        baseline_recalls = [q["baseline"]["accuracy"]["number_recall"] for q in results["questions"]]
        rag_recalls = [q["rag"]["accuracy"]["number_recall"] for q in results["questions"]]
        
        baseline_f1s = [q["baseline"]["accuracy"]["number_f1"] for q in results["questions"]]
        rag_f1s = [q["rag"]["accuracy"]["number_f1"] for q in results["questions"]]
        
        # LLM Judge results
        baseline_llm_correct = sum(1 for q in results["questions"] if q["baseline"]["accuracy"]["llm_judgment"]["is_correct"])
        rag_llm_correct = sum(1 for q in results["questions"] if q["rag"]["accuracy"]["llm_judgment"]["is_correct"])
        
        baseline_avg_confidence = sum(q["baseline"]["accuracy"]["llm_judgment"]["confidence"] for q in results["questions"]) / len(results["questions"])
        rag_avg_confidence = sum(q["rag"]["accuracy"]["llm_judgment"]["confidence"] for q in results["questions"]) / len(results["questions"])
        
        results["summary"] = {
            "baseline": {
                "avg_response_time": sum(baseline_times) / len(baseline_times),
                "avg_number_recall": sum(baseline_recalls) / len(baseline_recalls),
                "avg_number_f1": sum(baseline_f1s) / len(baseline_f1s),
                "perfect_answers": sum(1 for q in results["questions"] if q["baseline"]["accuracy"]["contains_all_numbers"]),
                "llm_correct_answers": baseline_llm_correct,
                "llm_avg_confidence": baseline_avg_confidence
            },
            "rag": {
                "avg_response_time": sum(rag_times) / len(rag_times),
                "avg_number_recall": sum(rag_recalls) / len(rag_recalls),
                "avg_number_f1": sum(rag_f1s) / len(rag_f1s),
                "perfect_answers": sum(1 for q in results["questions"] if q["rag"]["accuracy"]["contains_all_numbers"]),
                "avg_sources_used": sum(q["rag"]["num_sources"] for q in results["questions"]) / len(results["questions"]),
                "llm_correct_answers": rag_llm_correct,
                "llm_avg_confidence": rag_avg_confidence
            }
        }
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*80}")
        logger.info("Evaluation completed!")
        logger.info(f"Results saved to: {output_path}")
        logger.info(f"{'='*80}")
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """Print summary of results."""
        summary = results["summary"]
        total = results['metadata']['num_questions']
        
        print("\n" + "="*80)
        print("FACTUAL PRECISION EVALUATION SUMMARY")
        print("="*80)
        
        print("\nBaseline (Non-RAG):")
        print(f"  Average Response Time: {summary['baseline']['avg_response_time']:.2f}s")
        print(f"  Average Number Recall: {summary['baseline']['avg_number_recall']:.2%}")
        print(f"  Average Number F1: {summary['baseline']['avg_number_f1']:.2%}")
        print(f"  Perfect Answers (All Numbers): {summary['baseline']['perfect_answers']}/{total} ({summary['baseline']['perfect_answers']/total:.1%})")
        print(f"  LLM-Judged Correct: {summary['baseline']['llm_correct_answers']}/{total} ({summary['baseline']['llm_correct_answers']/total:.1%})")
        print(f"  LLM Judge Avg Confidence: {summary['baseline']['llm_avg_confidence']:.1%}")
        
        print("\nRAG System:")
        print(f"  Average Response Time: {summary['rag']['avg_response_time']:.2f}s")
        print(f"  Average Number Recall: {summary['rag']['avg_number_recall']:.2%}")
        print(f"  Average Number F1: {summary['rag']['avg_number_f1']:.2%}")
        print(f"  Perfect Answers (All Numbers): {summary['rag']['perfect_answers']}/{total} ({summary['rag']['perfect_answers']/total:.1%})")
        print(f"  LLM-Judged Correct: {summary['rag']['llm_correct_answers']}/{total} ({summary['rag']['llm_correct_answers']/total:.1%})")
        print(f"  LLM Judge Avg Confidence: {summary['rag']['llm_avg_confidence']:.1%}")
        print(f"  Average Sources Used: {summary['rag']['avg_sources_used']:.1f}")
        
        print("\nAccuracy Improvement (RAG vs Baseline):")
        recall_improvement = (summary['rag']['avg_number_recall'] - summary['baseline']['avg_number_recall'])
        print(f"  Number Recall: {recall_improvement:+.2%}")
        print(f"  Perfect Answers: {summary['rag']['perfect_answers'] - summary['baseline']['perfect_answers']:+d}")
        print(f"  LLM-Judged Correct: {summary['rag']['llm_correct_answers'] - summary['baseline']['llm_correct_answers']:+d}")
        
        print("\n" + "="*80)
        print(f"Evaluation Method: LLM-as-a-Judge using {results['metadata']['llm_model']} (temperature=0.0)")
        print("="*80 + "\n")


def main():
    """Main execution."""
    print("=" * 80)
    print("STARTING FACTUAL PRECISION EVALUATION")
    print("=" * 80 + "\n")
    
    questions_path = Path(__file__).parent / "factual_questions.json"
    output_path = Path(__file__).parent / f"factual_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    print(f"Questions file: {questions_path}")
    print(f"Output file: {output_path}")
    print(f"Questions file exists: {questions_path.exists()}\n")
    
    try:
        print("Initializing evaluator...")
        evaluator = FactualEvaluator()
        
        print("Loading questions...")
        questions = evaluator.load_questions(questions_path)
        
        print(f"\nStarting evaluation of {len(questions)} questions...")
        print("This will take approximately 10-15 minutes due to LLM-as-a-Judge calls.\n")
        
        results = evaluator.evaluate(questions, output_path)
        evaluator.print_summary(results)
        
        print(f"\nNext: Run 'python visualize_factual.py' to generate charts")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nMake sure you have:")
        print("1. Run 'python main.py --index' to create the vector store")
        print("2. The factual_questions.json file exists")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

