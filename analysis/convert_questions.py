"""
Convert question.txt to JSON format for evaluation.
"""

import json
from pathlib import Path

def parse_questions(txt_file):
    """Parse question.txt into structured format."""
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    questions = []
    blocks = content.split('--------------------------------------------')
    
    for block in blocks:
        if not block.strip() or 'RAG EVALUATION DATASET' in block:
            continue
        
        lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        
        if len(lines) < 3:
            continue
        
        # Parse question number
        question_id = None
        question_text = None
        answer_text = None
        source_text = None
        
        i = 0
        while i < len(lines):
            if lines[i].endswith('.') and lines[i][:-1].isdigit():
                question_id = int(lines[i][:-1])
            elif lines[i] == 'Question:':
                question_text = lines[i+1] if i+1 < len(lines) else None
                i += 1
            elif lines[i] == 'Answer:':
                answer_text = lines[i+1] if i+1 < len(lines) else None
                i += 1
            elif lines[i] == 'Source:':
                source_text = lines[i+1] if i+1 < len(lines) else None
                i += 1
            i += 1
        
        if question_id and question_text and answer_text and source_text:
            # Determine category based on source
            if 'iSAID' in source_text:
                category = 'dataset_facts_isaid'
            elif 'OpenEarthMap' in source_text or 'openearthmap' in source_text:
                category = 'dataset_facts_oem'
            elif 'Panoptic-DeepLab' in source_text:
                category = 'method_metrics_deeplab'
            elif 'SegFormer' in source_text:
                category = 'method_metrics_segformer'
            elif 'Mask2Former' in source_text:
                category = 'method_facts_mask2former'
            else:
                category = 'other_facts'
            
            questions.append({
                'id': question_id,
                'question': question_text,
                'ground_truth': answer_text,
                'source': source_text,
                'category': category
            })
    
    return questions

def main():
    input_file = Path(__file__).parent / 'question.txt'
    output_file = Path(__file__).parent / 'factual_questions.json'
    
    questions = parse_questions(input_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(questions)} questions to {output_file}")
    
    # Print summary
    categories = {}
    for q in questions:
        cat = q['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} questions")

if __name__ == '__main__':
    main()

