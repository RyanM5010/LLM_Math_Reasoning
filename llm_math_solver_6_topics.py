"""
Math Problem Solver with LLM and Evaluation Framework
====================================================

This module provides a complete pipeline for:
1. Loading math datasets from DeepMind Math Dataset
2. Solving math problems using LLM (OpenAI GPT-4)
3. Evaluating answer accuracy using symbolic comparison

Author: Rui Min
"""

import pandas as pd
import numpy as np
import os
import re
import time
from typing import List, Tuple, Optional
from sympy import simplify, sympify, Eq
from datasets import load_dataset
from openai import OpenAI
from google.colab import userdata

# ===== Configuration =====

# Topic list for math dataset
TOPICS = [
    "algebra__polynomial_roots",
    "algebra__sequence_nth_term", 
    "arithmetic__mul_div_multiple",
    "arithmetic__simplify_surd",
    "comparison__sort_composed",
    "numbers__base_conversion"
]

SAMPLES_PER_TOPIC = 500

# ===== Prompt Templates =====

ORIGINAL_PROMPT = """I want to create a mathematical problem solver. The AI will be given a single math question and must solve it internally, but only return the final answer. It should not show intermediate steps, explanations, or reasoning. For example, if the answer is x = 5, return: 5. If the answer is a list, return it like [1, 2, 3]. No extra text. Just the final result."""

META_PROMPT = """You are a mathematical problem solver. You will be given a single math question to solve. Your task is to solve the problem internally and return only the final answer, without showing any intermediate steps, explanations, or reasoning.

Here is the math question:
<question>
{{MATH_QUESTION}}
</question>

Solve this problem internally. Do not show any work or explain your process. Once you have the final answer, return it in the following format:

- If the answer is a single number or variable, simply return that value. For example: 5 or x
- If the answer is an equation, return it without any additional text. For example: y = 2x + 3
- If the answer is a list or set, return it in square brackets with comma-separated values. For example: [1, 2, 3] or [-1, 0, 1]

Do not include any additional text, explanations, or formatting. Your entire response should consist solely of the final result in one of the formats described above.

Examples of correct output:
7
x = 10
[-2, 0, 2]
y = x^2 - 4

Provide your final answer now:"""

NUMERICAL_PROMPT = """You are a mathematical problem solver. You will be given a single math question to solve. Your task is to return only the final answer, without showing any intermediate steps, explanations, or reasoning.

Here is the math question:
<question>
{{MATH_QUESTION}}
</question>

Solve this problem internally. If the problem includes variables (e.g., w, x), and the final answer depends on or refers to them, **do not substitute numerical values**. Return symbolic results that preserve variable names.

Return your result in the following formats:
- For a number or variable: 5 or x
- For an equation: y = 2x + 3
- For a list/set: [x, y, z] or [a + b, 2a]
- For symbolic expressions: f^2 + 33f + 2264
- For ordering/comparison questions: return symbols or expressions as-is (e.g., 112, w, 1), **not numerical values**

Do not include any extra text, explanations, or formatting. Your entire response should consist only of the final result.

Examples of correct output:
7
x = 10
[-2, 0, 2]
y = x^2 - 4
f^2 + 33f + 2264
112, w, 1

Provide your final answer now:"""

# ===== Setup Functions =====

def setup_openai_client():
    """Initialize OpenAI client with API key from Colab secrets"""
    api_key = userdata.get('openai_api')
    if api_key is None:
        raise ValueError("Missing secret: openai_api")
    return OpenAI(api_key=api_key)

def install_requirements():
    """Install required packages"""
    os.system("pip install --upgrade datasets openai sympy")

# ===== Data Loading Functions =====

def load_math_dataset_topics(topics: List[str], samples_per_topic: int = 500, output_dir: str = "datasets") -> None:
    """
    Load specified topics from DeepMind Math Dataset and save as CSV files
    
    Args:
        topics: List of topic names to load
        samples_per_topic: Number of samples to load per topic
        output_dir: Directory to save CSV files
    """
    print("Loading DeepMind Math Dataset...")
    
    # Load the full dataset
    dataset = load_dataset("deepmind/math_dataset", split="train")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for topic in topics:
        print(f"Processing topic: {topic}")
        
        # Filter data for specific topic
        df_topic = dataset.filter(lambda x: x["category"] == topic).select(range(samples_per_topic)).to_pandas()
        df_topic['topic'] = topic
        
        # Save to CSV
        filename = f"{output_dir}/{topic.replace('__', '_')}.csv"
        df_topic[['question', 'answer', 'topic']].to_csv(filename, index=False)
        print(f"✅ Saved {filename}, shape={df_topic.shape}")

def load_single_topic(topic: str, samples: int = 500) -> pd.DataFrame:
    """Load a single topic for quick testing"""
    dataset = load_dataset("deepmind/math_dataset", topic, split="train")
    df = pd.DataFrame(dataset[:samples])
    df['topic'] = topic
    return df[['question', 'answer', 'topic']]

# ===== LLM Solving Functions =====

def get_numerical_answer(question: str, client: OpenAI, prompt_template: str = NUMERICAL_PROMPT, 
                        retries: int = 3, delay: int = 2) -> str:
    """
    Get numerical answer from LLM for a math question
    
    Args:
        question: Math question to solve
        client: OpenAI client instance
        prompt_template: Prompt template to use
        retries: Number of retry attempts
        delay: Delay between retries
        
    Returns:
        LLM response or error message
    """
    for attempt in range(retries):
        try:
            # Replace placeholder with actual question
            prompt = prompt_template.replace("{{MATH_QUESTION}}", question)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                print(f"Retrying ({attempt + 1}/{retries}) after error: {e}")
                time.sleep(delay)
            else:
                return f"ERROR: {str(e)}"

def process_file(file_name: str, client: OpenAI, input_dir: str = "./", 
                output_dir: str = "./numerical_outputs", save_every: int = 10, 
                sleep_time: int = 1, prompt_template: str = NUMERICAL_PROMPT) -> None:
    """
    Process entire CSV file and get LLM answers for all questions
    
    Args:
        file_name: Name of input CSV file
        client: OpenAI client instance
        input_dir: Input directory path
        output_dir: Output directory path
        save_every: Save progress every N questions
        sleep_time: Sleep time between API calls
        prompt_template: Prompt template to use
    """
    os.makedirs(output_dir, exist_ok=True)
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name.replace(".csv", "_numerical.csv"))
    
    df = pd.read_csv(input_path)
    answers = []
    
    print(f"Processing {len(df)} questions from {file_name}...")
    
    for idx, row in df.iterrows():
        question = row['question']
        print(f"[{file_name}] Processing {idx + 1}/{len(df)}: {question[:60]}...")
        answer = get_numerical_answer(question, client, prompt_template)
        answers.append(answer)
        
        # Save progress periodically
        if (idx + 1) % save_every == 0:
            temp_df = df.copy()
            temp_df['numerical_answer'] = pd.Series(answers + [''] * (len(df) - len(answers)))
            temp_df.to_csv(output_path, index=False)
            print(f"Progress saved: {idx + 1} questions")
        
        time.sleep(sleep_time)
    
    # Final save
    df['numerical_answer'] = answers
    df.to_csv(output_path, index=False)
    print(f"✅ Completed: {file_name}")

# ===== Evaluation Functions =====

def clean_answer(answer):
    """Clean and convert answer to appropriate format"""
    if isinstance(answer, str):
        cleaned = answer.replace("b'", "").replace("\\n'", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return cleaned
    return answer

def is_equivalent(expr1, expr2) -> bool:
    """
    Comprehensive comparison of mathematical expressions or answers.
    Compatible with strings, boolean values, equations, lists, symbolic expressions, and numerical answers.
    """
    try:
        # Step 1: Basic string cleaning
        str1 = str(expr1).strip()
        str2 = str(expr2).strip()
        
        # Step 2: Quick direct comparison
        if str1 == str2:
            return True
        
        # Step 3: Compare without spaces
        str1_norm = re.sub(r'\s+', '', str1)
        str2_norm = re.sub(r'\s+', '', str2)
        if str1_norm == str2_norm:
            return True
        
        # Step 4: Boolean value comparison
        bool_map = {
            'true': True, 'false': False, 'yes': True, 'no': False,
            '1': True, '0': False, 't': True, 'f': False
        }
        if str1.lower() in bool_map and str2.lower() in bool_map:
            return bool_map[str1.lower()] == bool_map[str2.lower()]
        
        # Step 5: List comparison (ordered)
        if str1.startswith('[') and str1.endswith(']') and str2.startswith('[') and str2.endswith(']'):
            try:
                list1 = [x.strip() for x in str1[1:-1].split(',') if x.strip()]
                list2 = [x.strip() for x in str2[1:-1].split(',') if x.strip()]
                if len(list1) != len(list2):
                    return False
                for a, b in zip(list1, list2):
                    if not is_equivalent(a, b):
                        return False
                return True
            except Exception:
                pass
        
        # Step 6: Equation comparison (left-right sides)
        if '=' in str1 and '=' in str2:
            try:
                l1, r1 = str1.split('=', 1)
                l2, r2 = str2.split('=', 1)
                # Consider left-right swap
                if (is_equivalent(l1, l2) and is_equivalent(r1, r2)) or \
                   (is_equivalent(l1, r2) and is_equivalent(r1, l2)):
                    return True
            except Exception:
                pass
        
        # Step 7: Format normalization
        def normalize_expr(s):
            s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)  # 2x -> 2*x
            s = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', s)  # xy -> x*y
            s = s.replace('^', '**')
            return s
        
        norm1 = normalize_expr(str1_norm)
        norm2 = normalize_expr(str2_norm)
        
        # Step 8: Sympy symbolic comparison
        try:
            sym1 = simplify(sympify(norm1, strict=False))
            sym2 = simplify(sympify(norm2, strict=False))
            eq_result = Eq(sym1, sym2)
            # Convert Eq result to boolean
            if hasattr(eq_result, 'equals'):
                return eq_result.equals(True)
            else:
                return bool(eq_result)
        except Exception:
            pass
        
        # Step 9: Simple numerical comparison
        try:
            if all(c in '0123456789.+-*/()' for c in norm1 + norm2):
                val1 = eval(norm1)
                val2 = eval(norm2)
                if abs(val1 - val2) < 1e-10:
                    return True
        except Exception:
            pass
        
        # Step 10: Final fallback
        return norm1 == norm2
        
    except Exception:
        # Complete failure fallback
        return str1 == str2

def calculate_topic_accuracy(numerical_answers_path: str, questions_path: str, 
                           output_dir: str = "./comparison_results") -> Tuple[float, int, int]:
    """
    Compare model predictions vs ground truth answers for a single topic file.
    Uses sympy to check expression equivalence and saves comparison results.
    
    Args:
        numerical_answers_path: Path to file with LLM predictions
        questions_path: Path to file with ground truth answers
        output_dir: Directory to save comparison results
        
    Returns:
        Tuple of (accuracy, correct_count, total_count)
    """
    print(f"\nEvaluating: {numerical_answers_path}")
    
    df_pred = pd.read_csv(numerical_answers_path)
    df_gt = pd.read_csv(questions_path)
    
    # Clean answers
    df_pred['numerical_answer'] = df_pred['numerical_answer'].apply(clean_answer)
    df_gt['answer_cleaned'] = df_gt['answer'].apply(clean_answer)
    
    # Merge on question to align predictions and ground truth
    df = pd.merge(df_pred, df_gt[['question', 'answer_cleaned']], on='question', how='inner')
    
    matches = []
    for _, row in df.iterrows():
        pred_answer = row['numerical_answer']
        true_answer = row['answer_cleaned']
        
        if pd.isna(pred_answer) or pd.isna(true_answer):
            match = False
        else:
            try:
                match = is_equivalent(pred_answer, true_answer)
            except Exception:
                match = str(pred_answer).strip() == str(true_answer).strip()
        matches.append(bool(match))
    
    df['match'] = matches
    correct = sum(matches)
    total = len(matches)
    accuracy = round(correct / total * 100, 2) if total > 0 else 0.0
    
    print(f"Accuracy: {accuracy}% ({correct}/{total})")
    
    # Save comparison results
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(numerical_answers_path).replace(".csv", "_comparison.csv")
    output_path = os.path.join(output_dir, base_name)
    df.to_csv(output_path, index=False)
    print(f"✅ Comparison saved to: {output_path}")
    
    return accuracy, correct, total

# ===== Main Pipeline Functions =====

def run_complete_pipeline(topics: List[str] = TOPICS, samples_per_topic: int = SAMPLES_PER_TOPIC,
                         prompt_template: str = NUMERICAL_PROMPT) -> None:
    """
    Run the complete pipeline: data loading -> solving -> evaluation
    
    Args:
        topics: List of topics to process
        samples_per_topic: Number of samples per topic
        prompt_template: Prompt template to use for solving
    """
    print("=== Math Problem Solver Pipeline ===")
    
    # Step 1: Install requirements
    print("\n1. Installing requirements...")
    install_requirements()
    
    # Step 2: Setup OpenAI client
    print("\n2. Setting up OpenAI client...")
    client = setup_openai_client()
    
    # Step 3: Load dataset
    print("\n3. Loading math dataset...")
    load_math_dataset_topics(topics, samples_per_topic)
    
    # Step 4: Process files with LLM
    print("\n4. Processing files with LLM...")
    for topic in topics:
        filename = f"{topic.replace('__', '_')}.csv"
        process_file(filename, client, input_dir="datasets", prompt_template=prompt_template)
    
    # Step 5: Evaluate results
    print("\n5. Evaluating results...")
    results = {}
    for topic in topics:
        filename = f"{topic.replace('__', '_')}.csv"
        pred_file = f"numerical_outputs/{filename.replace('.csv', '_numerical.csv')}"
        gt_file = f"datasets/{filename}"
        
        accuracy, correct, total = calculate_topic_accuracy(pred_file, gt_file)
        results[topic] = {'accuracy': accuracy, 'correct': correct, 'total': total}
    
    # Step 6: Summary
    print("\n=== FINAL RESULTS ===")
    total_correct = sum(r['correct'] for r in results.values())
    total_questions = sum(r['total'] for r in results.values())
    overall_accuracy = round(total_correct / total_questions * 100, 2) if total_questions > 0 else 0.0
    
    for topic, result in results.items():
        print(f"{topic}: {result['accuracy']}% ({result['correct']}/{result['total']})")
    
    print(f"\nOverall Accuracy: {overall_accuracy}% ({total_correct}/{total_questions})")

def quick_test(topic: str = "algebra__polynomial_roots", num_questions: int = 5) -> None:
    """Quick test with a small number of questions"""
    print(f"=== Quick Test: {topic} ({num_questions} questions) ===")
    
    client = setup_openai_client()
    
    # Load small sample
    df = load_single_topic(topic, num_questions)
    
    # Process questions
    answers = []
    for idx, row in df.iterrows():
        question = row['question']
        print(f"Q{idx+1}: {question}")
        answer = get_numerical_answer(question, client)
        answers.append(answer)
        print(f"A{idx+1}: {answer}")
        print(f"Expected: {row['answer']}")
        print("-" * 50)
    
    df['predicted_answer'] = answers
    return df

# ===== Example Usage =====

if __name__ == "__main__":
    # Quick test
    # quick_test("algebra__polynomial_roots", 3)
    
    # Full pipeline
    run_complete_pipeline()