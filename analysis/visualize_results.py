"""
Visualization script for RAG evaluation results.
Creates charts and plots comparing RAG vs non-RAG performance.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime

# Matplotlib configuration for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


class ResultsVisualizer:
    """
    Visualizes RAG evaluation results with various charts and plots.
    """
    
    def __init__(self, results_path: Path):
        """
        Initialize visualizer with results data.
        
        Args:
            results_path: Path to evaluation results JSON file
        """
        with open(results_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        self.output_dir = results_path.parent / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Loaded results from: {results_path}")
        print(f"Visualizations will be saved to: {self.output_dir}")
    
    def plot_response_time_comparison(self):
        """Create bar chart comparing response times."""
        questions = self.results["questions"]
        question_ids = [q["id"] for q in questions]
        baseline_times = [q["baseline"]["response_time"] for q in questions]
        rag_times = [q["rag"]["response_time"] for q in questions]
        
        x = np.arange(len(question_ids))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - width/2, baseline_times, width, label='Baseline (No RAG)', 
                       color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, rag_times, width, label='RAG System', 
                       color='#4ECDC4', alpha=0.8)
        
        ax.set_xlabel('Question ID', fontweight='bold')
        ax.set_ylabel('Response Time (seconds)', fontweight='bold')
        ax.set_title('Response Time Comparison: RAG vs Baseline', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(question_ids)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add average lines
        avg_baseline = np.mean(baseline_times)
        avg_rag = np.mean(rag_times)
        ax.axhline(y=avg_baseline, color='#FF6B6B', linestyle='--', linewidth=2, alpha=0.5, label=f'Baseline Avg: {avg_baseline:.2f}s')
        ax.axhline(y=avg_rag, color='#4ECDC4', linestyle='--', linewidth=2, alpha=0.5, label=f'RAG Avg: {avg_rag:.2f}s')
        
        plt.tight_layout()
        output_path = self.output_dir / "response_time_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_answer_length_comparison(self):
        """Create bar chart comparing answer lengths."""
        questions = self.results["questions"]
        question_ids = [q["id"] for q in questions]
        baseline_lengths = [q["baseline"]["answer_length"] for q in questions]
        rag_lengths = [q["rag"]["answer_length"] for q in questions]
        
        x = np.arange(len(question_ids))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars1 = ax.bar(x - width/2, baseline_lengths, width, label='Baseline (No RAG)', 
                       color='#FFB84D', alpha=0.8)
        bars2 = ax.bar(x + width/2, rag_lengths, width, label='RAG System', 
                       color='#A78BFA', alpha=0.8)
        
        ax.set_xlabel('Question ID', fontweight='bold')
        ax.set_ylabel('Answer Length (characters)', fontweight='bold')
        ax.set_title('Answer Length Comparison: RAG vs Baseline', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(question_ids)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "answer_length_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_summary_metrics(self):
        """Create summary comparison chart."""
        summary = self.results["summary"]
        
        metrics = ['Avg Response\nTime (s)', 'Avg Answer\nLength (chars)']
        baseline_values = [
            summary['baseline']['avg_response_time'],
            summary['baseline']['avg_answer_length']
        ]
        rag_values = [
            summary['rag']['avg_response_time'],
            summary['rag']['avg_answer_length']
        ]
        
        # Normalize values for better visualization
        baseline_norm = [baseline_values[0], baseline_values[1] / 100]
        rag_norm = [rag_values[0], rag_values[1] / 100]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Response Time
        bars1 = ax1.bar(['Baseline', 'RAG'], 
                        [baseline_values[0], rag_values[0]], 
                        color=['#FF6B6B', '#4ECDC4'], alpha=0.8, width=0.6)
        ax1.set_ylabel('Time (seconds)', fontweight='bold')
        ax1.set_title('Average Response Time', fontweight='bold', pad=15)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontweight='bold')
        
        # Answer Length
        bars2 = ax2.bar(['Baseline', 'RAG'], 
                        [baseline_values[1], rag_values[1]], 
                        color=['#FFB84D', '#A78BFA'], alpha=0.8, width=0.6)
        ax2.set_ylabel('Characters', fontweight='bold')
        ax2.set_title('Average Answer Length', fontweight='bold', pad=15)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Performance Summary: RAG vs Baseline', fontweight='bold', fontsize=16, y=1.02)
        plt.tight_layout()
        output_path = self.output_dir / "summary_metrics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_category_performance(self):
        """Create performance breakdown by question category."""
        questions = self.results["questions"]
        
        # Group by category
        categories = {}
        for q in questions:
            cat = q["category"]
            if cat not in categories:
                categories[cat] = {"baseline_times": [], "rag_times": [], "rag_sources": []}
            
            categories[cat]["baseline_times"].append(q["baseline"]["response_time"])
            categories[cat]["rag_times"].append(q["rag"]["response_time"])
            categories[cat]["rag_sources"].append(q["rag"]["num_sources"])
        
        # Calculate averages
        cat_names = list(categories.keys())
        baseline_avg = [np.mean(categories[cat]["baseline_times"]) for cat in cat_names]
        rag_avg = [np.mean(categories[cat]["rag_times"]) for cat in cat_names]
        sources_avg = [np.mean(categories[cat]["rag_sources"]) for cat in cat_names]
        
        x = np.arange(len(cat_names))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Response time by category
        bars1 = ax1.bar(x - width/2, baseline_avg, width, label='Baseline', 
                        color='#FF6B6B', alpha=0.8)
        bars2 = ax1.bar(x + width/2, rag_avg, width, label='RAG', 
                        color='#4ECDC4', alpha=0.8)
        
        ax1.set_xlabel('Question Category', fontweight='bold')
        ax1.set_ylabel('Avg Response Time (seconds)', fontweight='bold')
        ax1.set_title('Response Time by Category', fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(cat_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Sources used by category
        bars3 = ax2.bar(cat_names, sources_avg, color='#A78BFA', alpha=0.8)
        ax2.set_xlabel('Question Category', fontweight='bold')
        ax2.set_ylabel('Avg Sources Retrieved', fontweight='bold')
        ax2.set_title('Average Sources Retrieved by RAG System', fontweight='bold', pad=15)
        ax2.set_xticklabels(cat_names, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / "category_performance.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def plot_source_distribution(self):
        """Create chart showing distribution of sources used."""
        questions = self.results["questions"]
        source_counts = [q["rag"]["num_sources"] for q in questions]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(source_counts, bins=range(0, max(source_counts)+2), 
                color='#4ECDC4', alpha=0.8, edgecolor='black')
        ax1.set_xlabel('Number of Sources', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Distribution of Sources Retrieved', fontweight='bold', pad=15)
        ax1.grid(axis='y', alpha=0.3)
        
        # Line plot over questions
        ax2.plot(range(1, len(source_counts)+1), source_counts, 
                marker='o', linewidth=2, markersize=8, color='#4ECDC4', alpha=0.8)
        ax2.set_xlabel('Question ID', fontweight='bold')
        ax2.set_ylabel('Number of Sources', fontweight='bold')
        ax2.set_title('Sources Retrieved per Question', fontweight='bold', pad=15)
        ax2.grid(alpha=0.3)
        
        # Add average line
        avg_sources = np.mean(source_counts)
        ax2.axhline(y=avg_sources, color='#FF6B6B', linestyle='--', 
                   linewidth=2, alpha=0.5, label=f'Average: {avg_sources:.1f}')
        ax2.legend()
        
        plt.tight_layout()
        output_path = self.output_dir / "source_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def generate_statistics_report(self):
        """Generate a text report with detailed statistics."""
        summary = self.results["summary"]
        questions = self.results["questions"]
        
        report = []
        report.append("=" * 80)
        report.append("RAG EVALUATION STATISTICS REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Questions Evaluated: {len(questions)}")
        report.append(f"Model: {self.results['metadata']['llm_model']}")
        report.append(f"Embedding Model: {self.results['metadata']['embedding_model']}")
        
        report.append("\n" + "-" * 80)
        report.append("BASELINE SYSTEM (No RAG)")
        report.append("-" * 80)
        report.append(f"Average Response Time:  {summary['baseline']['avg_response_time']:.3f} seconds")
        report.append(f"Total Processing Time:  {summary['baseline']['total_time']:.2f} seconds")
        report.append(f"Average Answer Length:  {summary['baseline']['avg_answer_length']:.0f} characters")
        
        baseline_times = [q["baseline"]["response_time"] for q in questions]
        report.append(f"Min Response Time:      {min(baseline_times):.3f} seconds")
        report.append(f"Max Response Time:      {max(baseline_times):.3f} seconds")
        report.append(f"Std Dev Response Time:  {np.std(baseline_times):.3f} seconds")
        
        report.append("\n" + "-" * 80)
        report.append("RAG SYSTEM")
        report.append("-" * 80)
        report.append(f"Average Response Time:  {summary['rag']['avg_response_time']:.3f} seconds")
        report.append(f"Total Processing Time:  {summary['rag']['total_time']:.2f} seconds")
        report.append(f"Average Answer Length:  {summary['rag']['avg_answer_length']:.0f} characters")
        report.append(f"Average Sources Used:   {summary['rag']['avg_sources_used']:.2f} documents")
        
        rag_times = [q["rag"]["response_time"] for q in questions]
        report.append(f"Min Response Time:      {min(rag_times):.3f} seconds")
        report.append(f"Max Response Time:      {max(rag_times):.3f} seconds")
        report.append(f"Std Dev Response Time:  {np.std(rag_times):.3f} seconds")
        
        report.append("\n" + "-" * 80)
        report.append("COMPARATIVE ANALYSIS")
        report.append("-" * 80)
        
        time_diff = summary['rag']['avg_response_time'] - summary['baseline']['avg_response_time']
        time_pct = (time_diff / summary['baseline']['avg_response_time']) * 100
        report.append(f"Response Time Difference: {time_diff:+.3f} seconds ({time_pct:+.1f}%)")
        
        length_diff = summary['rag']['avg_answer_length'] - summary['baseline']['avg_answer_length']
        length_pct = (length_diff / summary['baseline']['avg_answer_length']) * 100
        report.append(f"Answer Length Difference: {length_diff:+.0f} characters ({length_pct:+.1f}%)")
        
        # Category breakdown
        report.append("\n" + "-" * 80)
        report.append("PERFORMANCE BY CATEGORY")
        report.append("-" * 80)
        
        categories = {}
        for q in questions:
            cat = q["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(q)
        
        for cat, cat_questions in sorted(categories.items()):
            report.append(f"\n{cat.upper()}:")
            report.append(f"  Questions: {len(cat_questions)}")
            
            cat_baseline_avg = np.mean([q["baseline"]["response_time"] for q in cat_questions])
            cat_rag_avg = np.mean([q["rag"]["response_time"] for q in cat_questions])
            cat_sources_avg = np.mean([q["rag"]["num_sources"] for q in cat_questions])
            
            report.append(f"  Baseline Avg Time: {cat_baseline_avg:.3f}s")
            report.append(f"  RAG Avg Time: {cat_rag_avg:.3f}s")
            report.append(f"  RAG Avg Sources: {cat_sources_avg:.2f}")
        
        report.append("\n" + "=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        output_path = self.output_dir / "statistics_report.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"✓ Saved: {output_path}")
        print("\n" + report_text)
    
    def generate_all_visualizations(self):
        """Generate all visualizations and reports."""
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80 + "\n")
        
        self.plot_response_time_comparison()
        self.plot_answer_length_comparison()
        self.plot_summary_metrics()
        self.plot_category_performance()
        self.plot_source_distribution()
        self.generate_statistics_report()
        
        print("\n" + "=" * 80)
        print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80 + "\n")


def main():
    """Main execution function."""
    # Find the most recent results file
    analysis_dir = Path(__file__).parent
    results_files = list(analysis_dir.glob("evaluation_results_*.json"))
    
    if not results_files:
        print("Error: No evaluation results found.")
        print("Please run 'python analysis/evaluate_rag.py' first.")
        sys.exit(1)
    
    # Use the most recent file
    latest_results = max(results_files, key=lambda p: p.stat().st_mtime)
    print(f"\nUsing results file: {latest_results.name}\n")
    
    # Create visualizer and generate plots
    visualizer = ResultsVisualizer(latest_results)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()

