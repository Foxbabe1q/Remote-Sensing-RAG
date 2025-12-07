"""
Visualization script for factual precision evaluation results.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class FactualResultsVisualizer:
    """Visualizes factual precision evaluation results."""
    
    def __init__(self, results_path: Path):
        with open(results_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        self.output_dir = results_path.parent / "factual_visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Loaded results from: {results_path}")
        print(f"Visualizations will be saved to: {self.output_dir}")
    
    def plot_accuracy_comparison(self):
        """Plot number recall accuracy comparison."""
        questions = self.results["questions"]
        question_ids = [q["id"] for q in questions]
        
        baseline_recalls = [q["baseline"]["accuracy"]["number_recall"] * 100 for q in questions]
        rag_recalls = [q["rag"]["accuracy"]["number_recall"] * 100 for q in questions]
        
        x = np.arange(len(question_ids))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        bars1 = ax.bar(x - width/2, baseline_recalls, width, label='Baseline (No RAG)', 
                       color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, rag_recalls, width, label='RAG System', 
                       color='#51CF66', alpha=0.8)
        
        ax.set_xlabel('Question ID', fontweight='bold')
        ax.set_ylabel('Number Recall (%)', fontweight='bold')
        ax.set_title('Factual Accuracy: Number Recall Comparison', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(question_ids)
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add 100% reference line
        ax.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.3, label='Perfect Accuracy')
        
        plt.tight_layout()
        output_path = self.output_dir / "accuracy_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_perfect_answers(self):
        """Plot perfect answer counts."""
        summary = self.results["summary"]
        total = self.results["metadata"]["num_questions"]
        
        baseline_perfect = summary["baseline"]["perfect_answers"]
        rag_perfect = summary["rag"]["perfect_answers"]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Baseline\n(No RAG)', 'RAG System']
        values = [baseline_perfect, rag_perfect]
        colors = ['#FF6B6B', '#51CF66']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8, width=0.5)
        
        ax.set_ylabel('Number of Perfect Answers', fontweight='bold')
        ax.set_title(f'Perfect Answers (All Numbers Correct) out of {total} Questions', 
                     fontweight='bold', pad=20)
        ax.set_ylim(0, total + 2)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(val)}\n({val/total*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add reference line for total
        ax.axhline(y=total, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(1.4, total, f'Total: {total}', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        output_path = self.output_dir / "perfect_answers.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_category_accuracy(self):
        """Plot accuracy by question category."""
        questions = self.results["questions"]
        
        categories = {}
        for q in questions:
            cat = q["category"]
            if cat not in categories:
                categories[cat] = {"baseline": [], "rag": []}
            
            categories[cat]["baseline"].append(q["baseline"]["accuracy"]["number_recall"] * 100)
            categories[cat]["rag"].append(q["rag"]["accuracy"]["number_recall"] * 100)
        
        cat_names = list(categories.keys())
        baseline_avg = [np.mean(categories[cat]["baseline"]) for cat in cat_names]
        rag_avg = [np.mean(categories[cat]["rag"]) for cat in cat_names]
        
        x = np.arange(len(cat_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, baseline_avg, width, label='Baseline', 
                       color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, rag_avg, width, label='RAG', 
                       color='#51CF66', alpha=0.8)
        
        ax.set_xlabel('Question Category', fontweight='bold')
        ax.set_ylabel('Average Number Recall (%)', fontweight='bold')
        ax.set_title('Accuracy by Question Category', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(cat_names, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "category_accuracy.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_summary_metrics(self):
        """Plot summary comparison."""
        summary = self.results["summary"]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Number Recall
        recalls = [summary['baseline']['avg_number_recall'] * 100, 
                   summary['rag']['avg_number_recall'] * 100]
        ax1.bar(['Baseline', 'RAG'], recalls, color=['#FF6B6B', '#51CF66'], alpha=0.8, width=0.6)
        ax1.set_ylabel('Recall (%)', fontweight='bold')
        ax1.set_title('Average Number Recall', fontweight='bold')
        ax1.set_ylim(0, 105)
        ax1.grid(axis='y', alpha=0.3)
        for i, v in enumerate(recalls):
            ax1.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # F1 Score
        f1s = [summary['baseline']['avg_number_f1'] * 100,
               summary['rag']['avg_number_f1'] * 100]
        ax2.bar(['Baseline', 'RAG'], f1s, color=['#FF6B6B', '#51CF66'], alpha=0.8, width=0.6)
        ax2.set_ylabel('F1 Score (%)', fontweight='bold')
        ax2.set_title('Average Number F1 Score', fontweight='bold')
        ax2.set_ylim(0, 105)
        ax2.grid(axis='y', alpha=0.3)
        for i, v in enumerate(f1s):
            ax2.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Response Time
        times = [summary['baseline']['avg_response_time'],
                 summary['rag']['avg_response_time']]
        ax3.bar(['Baseline', 'RAG'], times, color=['#FF6B6B', '#51CF66'], alpha=0.8, width=0.6)
        ax3.set_ylabel('Time (seconds)', fontweight='bold')
        ax3.set_title('Average Response Time', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        for i, v in enumerate(times):
            ax3.text(i, v, f'{v:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # Perfect Answers
        perfects = [summary['baseline']['perfect_answers'],
                    summary['rag']['perfect_answers']]
        ax4.bar(['Baseline', 'RAG'], perfects, color=['#FF6B6B', '#51CF66'], alpha=0.8, width=0.6)
        ax4.set_ylabel('Count', fontweight='bold')
        ax4.set_title('Perfect Answers (All Numbers Correct)', fontweight='bold')
        ax4.set_ylim(0, self.results['metadata']['num_questions'] + 2)
        ax4.grid(axis='y', alpha=0.3)
        for i, v in enumerate(perfects):
            ax4.text(i, v, f'{int(v)}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Factual Precision Evaluation Summary', fontweight='bold', fontsize=16, y=1.00)
        plt.tight_layout()
        output_path = self.output_dir / "summary_metrics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def generate_statistics_report(self):
        """Generate text report."""
        summary = self.results["summary"]
        metadata = self.results["metadata"]
        questions = self.results["questions"]
        
        report = []
        report.append("=" * 80)
        report.append("FACTUAL PRECISION EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Questions: {metadata['num_questions']}")
        report.append(f"Model: {metadata['llm_model']}")
        report.append(f"Evaluation Type: {metadata['evaluation_type']}")
        
        report.append("\n" + "-" * 80)
        report.append("BASELINE SYSTEM (No RAG)")
        report.append("-" * 80)
        report.append(f"Average Response Time:  {summary['baseline']['avg_response_time']:.3f} seconds")
        report.append(f"Average Number Recall:  {summary['baseline']['avg_number_recall']:.2%}")
        report.append(f"Average Number F1:      {summary['baseline']['avg_number_f1']:.2%}")
        report.append(f"Perfect Answers:        {summary['baseline']['perfect_answers']}/{metadata['num_questions']} ({summary['baseline']['perfect_answers']/metadata['num_questions']:.1%})")
        
        report.append("\n" + "-" * 80)
        report.append("RAG SYSTEM")
        report.append("-" * 80)
        report.append(f"Average Response Time:  {summary['rag']['avg_response_time']:.3f} seconds")
        report.append(f"Average Number Recall:  {summary['rag']['avg_number_recall']:.2%}")
        report.append(f"Average Number F1:      {summary['rag']['avg_number_f1']:.2%}")
        report.append(f"Perfect Answers:        {summary['rag']['perfect_answers']}/{metadata['num_questions']} ({summary['rag']['perfect_answers']/metadata['num_questions']:.1%})")
        report.append(f"Average Sources Used:   {summary['rag']['avg_sources_used']:.2f}")
        
        report.append("\n" + "-" * 80)
        report.append("ACCURACY IMPROVEMENT (RAG vs Baseline)")
        report.append("-" * 80)
        recall_diff = summary['rag']['avg_number_recall'] - summary['baseline']['avg_number_recall']
        f1_diff = summary['rag']['avg_number_f1'] - summary['baseline']['avg_number_f1']
        perfect_diff = summary['rag']['perfect_answers'] - summary['baseline']['perfect_answers']
        
        report.append(f"Number Recall:  {recall_diff:+.2%}")
        report.append(f"Number F1:      {f1_diff:+.2%}")
        report.append(f"Perfect Answers: {perfect_diff:+d} ({perfect_diff/metadata['num_questions']:+.1%})")
        
        report.append("\n" + "-" * 80)
        report.append("CATEGORY BREAKDOWN")
        report.append("-" * 80)
        
        categories = {}
        for q in questions:
            cat = q["category"]
            if cat not in categories:
                categories[cat] = {"baseline": [], "rag": [], "count": 0}
            categories[cat]["baseline"].append(q["baseline"]["accuracy"]["number_recall"])
            categories[cat]["rag"].append(q["rag"]["accuracy"]["number_recall"])
            categories[cat]["count"] += 1
        
        for cat in sorted(categories.keys()):
            report.append(f"\n{cat.upper()}:")
            report.append(f"  Questions: {categories[cat]['count']}")
            report.append(f"  Baseline Avg Recall: {np.mean(categories[cat]['baseline']):.2%}")
            report.append(f"  RAG Avg Recall:      {np.mean(categories[cat]['rag']):.2%}")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        output_path = self.output_dir / "factual_statistics.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✓ Saved: {output_path}")
        print("\n" + report_text)
    
    def generate_all(self):
        """Generate all visualizations."""
        print("\n" + "=" * 80)
        print("GENERATING FACTUAL PRECISION VISUALIZATIONS")
        print("=" * 80 + "\n")
        
        self.plot_accuracy_comparison()
        self.plot_perfect_answers()
        self.plot_category_accuracy()
        self.plot_summary_metrics()
        self.generate_statistics_report()
        
        print("\n" + "=" * 80)
        print("ALL VISUALIZATIONS GENERATED")
        print(f"Output: {self.output_dir}")
        print("=" * 80 + "\n")


def main():
    """Main execution."""
    analysis_dir = Path(__file__).parent
    results_files = list(analysis_dir.glob("factual_evaluation_*.json"))
    
    if not results_files:
        print("Error: No factual evaluation results found.")
        print("Run 'python analysis/evaluate_factual.py' first.")
        sys.exit(1)
    
    latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
    print(f"\nUsing results: {latest_results.name}\n")
    
    visualizer = FactualResultsVisualizer(latest_results)
    visualizer.generate_all()


if __name__ == "__main__":
    main()

