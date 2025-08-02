"""
LangChain Math Problem Solver with Chain-of-Thought and Structured Output
=========================================================================

This module provides a comprehensive math problem solver using:
1. LangChain for orchestration
2. Pydantic for structured output parsing
3. Chain-of-Thought (CoT) prompting for reasoning
4. SymPy for symbolic mathematical comparison

Author: Rui Min
"""

import pandas as pd
import os
import time
import re
from typing import List, Optional, Dict, Any
from sympy import simplify, sympify, Eq

# LangChain and Pydantic imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# ===== Pydantic Models for Structured Output =====

class MathAnswerWithReasoning(BaseModel):
    """Structured output model for math problems with reasoning"""
    reasoning: str = Field(
        description="The step-by-step reasoning process used to arrive at the answer. "
                   "Explain the simplification and algebraic operations."
    )
    final_answer: str = Field(
        description="The final simplified mathematical expression or numerical value."
    )

# ===== System Prompts for Different Math Problem Types =====

SYSTEM_PROMPT_BASE_TEMPLATE = """You are an expert mathematical problem solver.
Your task is to solve the given math question. First, think step-by-step and write down your reasoning in the `reasoning` field.
Then, based on your reasoning, provide the final answer in the `final_answer` field.

Your entire response MUST be a single, valid JSON object that strictly adheres to the following schema.
{format_instructions}

RULES for the `final_answer` field:
- It must contain only the final mathematical result, derived from your reasoning.
- If the problem includes variables (e.g., w, x), return symbolic results.
- The format should be one of the following:
    - For a number or variable: 5 or x
    - For an equation: y = 2x + 3
    - For a list: [1, 2, 3]
    - For symbolic expressions: a*b + c

Now, solve the user's question and provide your response in the specified JSON format.
"""

SYSTEM_PROMPT_POLYNOMIAL = """You are an expert mathematical problem solver specializing in algebra. 
Your task is to carefully analyze the user's question and perform the requested operation. 
First, think step-by-step and write down your reasoning. Then, provide the final answer.

Your entire response MUST be a single, valid JSON object that strictly adheres to the following schema.
{format_instructions}

RULES for the `final_answer` field:
- If the question asks to "Solve for [variable]" or "Find the roots", the answer should be a Python-style list of numerical roots, e.g., `[-2, 0, 5]`.
- If the question asks to "Factor" an expression, the answer should be the fully factored symbolic expression, e.g., `(x-1)*(x+1)`.
- If the question asks to "Calculate [variable]" or "Solve for [variable] in terms of...", the answer should be a symbolic expression for that variable, e.g., `u = 9*t**(2/3)`.
- For other simplification tasks, provide the simplified expression.

--- EXAMPLES ---
Example 1 (Solving for roots):
Question: Solve for y: y**3 - y = 0.
{{
  "reasoning": "First, factor out the common term y, which gives y(y**2 - 1) = 0. This means one root is y=0. The second factor, y**2 - 1, is a difference of squares that factors into (y-1)(y+1). This gives the other two roots, y=1 and y=-1. The complete set of roots is -1, 0, and 1.",
  "final_answer": "[-1, 0, 1]"
}}

Example 2 (Factoring):
Question: Factor 2*k**2 - 2
{{
  "reasoning": "First, we can factor out the common constant 2, which gives 2*(k**2 - 1). The term k**2 - 1 is a difference of squares, which factors into (k-1)*(k+1). The fully factored expression is 2*(k-1)*(k+1).",
  "final_answer": "2*(k-1)*(k+1)"
}}
--- END OF EXAMPLES ---

Now, analyze the user's question and provide your response in the specified JSON format.
"""

SYSTEM_PROMPT_SEQUENCES = """You are an expert mathematical problem solver specializing in analyzing numerical sequences. 
Your task is to find a formula for the nth term of a given sequence. First, think step-by-step by analyzing the differences between the terms (first difference, second difference, etc.) to determine if the sequence is linear, quadratic, cubic, or other. Write down this analysis in the `reasoning` field. Then, provide the final formula in the `final_answer` field.

Your entire response MUST be a single, valid JSON object that strictly adheres to the following schema.
{format_instructions}

RULES for the `final_answer` field:
- The answer should be a symbolic expression in terms of the variable given in the question (e.g., n, u, w, etc.).
- The expression should correctly generate the terms of the sequence.

--- EXAMPLES ---
Example 1 (Linear Sequence):
Question: What is the nth term of 2, 5, 8, 11?
{{
  "reasoning": "The sequence is 2, 5, 8, 11. The first differences are 3, 3, 3. Since the first difference is constant, this is a linear (arithmetic) sequence with a common difference of 3. The formula is of the form a*n + b. The coefficient 'a' is the common difference, so a=3. For n=1, 3*1 + b = 2, so b = -1. The formula is 3*n - 1.",
  "final_answer": "3*n - 1"
}}

Example 2 (Quadratic Sequence):
Question: What is the wth term of 1, 4, 9, 16?
{{
  "reasoning": "The sequence is 1, 4, 9, 16. The first differences are 3, 5, 7. The second differences are 2, 2. Since the second difference is constant, this is a quadratic sequence. The formula is of the form a*w**2 + b*w + c. The coefficient 'a' is half the second difference, so a = 2/2 = 1. For w=1, 1**2 + b + c = 1. For w=2, 2**2 + 2b + c = 4. This gives b=0 and c=0. The formula is w**2.",
  "final_answer": "w**2"
}}
--- END OF EXAMPLES ---

Now, analyze the user's question, find the formula for the nth term, and provide your response in the specified JSON format.
"""

SYSTEM_PROMPT_ARITHMETIC = """You are an expert mathematical problem solver specializing in precise arithmetic calculations. 
Your task is to evaluate the given mathematical expression. First, think step-by-step, following the order of operations (PEMDAS/BODMAS), and write down this analysis in the `reasoning` field. Then, provide the final, simplified numerical answer in the `final_answer` field.

Your entire response MUST be a single, valid JSON object that strictly adheres to the following schema.
{format_instructions}

RULES for the `final_answer` field:
- The answer must be a single number.
- If the answer is not a whole number, present it as a simplified fraction (e.g., `a/b` or `-a/b`). Do not use decimals.

--- EXAMPLES ---
Example 1:
Question: Calculate (-1)*15/((1/12)/3/4)
{{
  "reasoning": "First, evaluate the denominator. (1/12)/3 is 1/36. Then, (1/36)/4 is 1/144. The expression becomes (-1)*15/(1/144). This is equivalent to -15 * 144, which is -2160.",
  "final_answer": "-2160"
}}

Example 2:
Question: Evaluate ((-47)/(-5217))/((-4)/24)
{{
  "reasoning": "First, simplify each fraction. (-47)/(-5217) simplifies to 47/5217, which is 1/111. (-4)/24 simplifies to -1/6. The expression becomes (1/111)/(-1/6). This is equivalent to (1/111) * (-6), which is -6/111. This fraction can be simplified by dividing the numerator and denominator by 3, resulting in -2/37.",
  "final_answer": "-2/37"
}}
--- END OF EXAMPLES ---

Now, analyze the user's question, evaluate the expression, and provide your response in the specified JSON format.
"""

SYSTEM_PROMPT_COMPARISON_SORT = """You are an expert mathematical problem solver specializing in multi-step algebraic reasoning and comparison. 
Your task is to carefully analyze the user's question, which involves several steps. First, think step-by-step and write down your reasoning. Then, provide the final answer.

Your entire response MUST be a single, valid JSON object that strictly adheres to the following schema.
{format_instructions}

Your reasoning process must be as follows:
1. Identify all variables and their definitions (e.g., `w = 0-3`). Calculate their numerical values.
2. Identify any functions and evaluate them as needed (e.g., `m(a) = -a - 2`, so `m(-10) = 8`).
3. Solve any equations to find the values of other variables (e.g., `9*t + 4 = o*t`).
4. Gather the final list of items to be sorted.
5. Sort these items according to the specified order (increasing or decreasing).
6. Construct the final answer string using the original variable names.

The `final_answer` field MUST be a single string of comma-separated values (e.g., `-7, t, w`).

--- EXAMPLE ---
Question: Let w = 0 - 3. Let m(a) = -a - 2. Let o be m(-10). Suppose 9*t + 4 = o*t. Sort w, t, -7 in increasing order.
{{
  "reasoning": "Step 1: Calculate w. w = 0 - 3 = -3. Step 2: Define m(a) = -a - 2. Step 3: Calculate o by evaluating m(-10). o = -(-10) - 2 = 10 - 2 = 8. Step 4: Solve for t using the value of o. 9*t + 4 = 8*t. Subtracting 9*t from both sides gives 4 = -t, so t = -4. Step 5: The items to sort are w, t, and -7, which correspond to the numerical values -3, -4, and -7. Step 6: Sorting these values in increasing order gives -7, -4, -3. Step 7: Replacing the numbers with their variable names gives the final sorted list: -7, t, w.",
  "final_answer": "-7, t, w"
}}
--- END OF EXAMPLE ---

Now, analyze the user's question, follow the reasoning steps, and provide your response in the specified JSON format.
"""

SYSTEM_PROMPT_BASE_CONVERSION = """You are an expert mathematical problem solver specializing in number base conversions.
Your task is to carefully analyze the user's question, which involves converting a number from a source base to a target base. First, think step-by-step and write down your reasoning. Then, provide the final answer.

Your entire response MUST be a single, valid JSON object that strictly adheres to the following schema.
{format_instructions}

Your reasoning process must be as follows:
1. Identify the number, the source base, and the target base from the question.
2. If the number is not in base 10, first convert it to base 10. Show the calculation (e.g., for '1A' in base 16, calculate 1*16^1 + 10*16^0 = 26).
3. Then, convert the base 10 number to the target base. Show the repeated division and remainder process.
4. Handle negative numbers correctly by converting the positive equivalent first, then prepending a negative sign.

The `final_answer` field MUST be a single string representing the number in the target base.

--- EXAMPLE ---
Question: What is -14744 (base 11) in base 10?
{{
  "reasoning": "Step 1: The number is -14744, the source base is 11, and the target base is 10. Step 2: First, I will convert the positive number 14744 (base 11) to base 10. The digits are 1, 4, 7, 4, 4. The calculation is 1*11^4 + 4*11^3 + 7*11^2 + 4*11^1 + 4*11^0 = 1*14641 + 4*1331 + 7*121 + 4*11 + 4*1 = 14641 + 5324 + 847 + 44 + 4 = 20860. Step 3: Since the original number was negative, the final answer in base 10 is -20860.",
  "final_answer": "-20860"
}}
--- END OF EXAMPLE ---

Now, analyze the user's question, follow the reasoning steps, and provide your response in the specified JSON format.
"""

SYSTEM_PROMPT_SIMPLIFY_SURD = """You are an expert mathematical problem solver specializing in simplifying surds and radical expressions.
Your task is to simplify the given expression involving square roots, radicals, and algebraic terms. First, think step-by-step and write down your reasoning. Then, provide the final simplified answer.

Your entire response MUST be a single, valid JSON object that strictly adheres to the following schema.
{format_instructions}

Your reasoning process should include:
1. Identify all radical terms and factor them if possible
2. Simplify square roots by extracting perfect squares
3. Combine like terms
4. Present the final simplified form

--- EXAMPLES ---
Example 1:
Question: b + sqrt(48)
{{
  "reasoning": "The term sqrt(48) can be simplified. 48 is 16 * 3. So, sqrt(48) is sqrt(16*3), which is 4*sqrt(3). The expression cannot be simplified further. The final expression is b + 4*sqrt(3).",
  "final_answer": "b + 4*sqrt(3)"
}}

Example 2:
Question: (a+b)**2 - (a-b)**2
{{
  "reasoning": "First, expand the terms. (a+b)**2 is a^2 + 2ab + b^2. (a-b)**2 is a^2 - 2ab + b^2. The expression becomes (a^2 + 2ab + b^2) - (a^2 - 2ab + b^2). This simplifies to a^2 + 2ab + b^2 - a^2 + 2ab - b^2. The a^2 and b^2 terms cancel out, leaving 2ab + 2ab, which is 4ab.",
  "final_answer": "4*a*b"
}}
--- END OF EXAMPLES ---

Now, analyze the user's question and provide your response in the specified JSON format.
"""

# ===== Prompt Templates Mapping =====

PROMPT_TEMPLATES = {
    "polynomial": SYSTEM_PROMPT_POLYNOMIAL,
    "sequences": SYSTEM_PROMPT_SEQUENCES,
    "arithmetic": SYSTEM_PROMPT_ARITHMETIC,
    "comparison_sort": SYSTEM_PROMPT_COMPARISON_SORT,
    "base_conversion": SYSTEM_PROMPT_BASE_CONVERSION,
    "simplify_surd": SYSTEM_PROMPT_SIMPLIFY_SURD,
    "default": SYSTEM_PROMPT_BASE_TEMPLATE
}

# ===== Setup and Configuration =====

def install_requirements():
    """Install required packages"""
    os.system("pip install -U langchain langchain-openai")

def setup_api_key():
    """Setup OpenAI API key from various sources"""
    api_key = None
    
    # Try Google Colab first
    try:
        from google.colab import userdata
        api_key = userdata.get('openai_api')
    except (ImportError, ModuleNotFoundError):
        pass
    
    # Try environment variable
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API Key not found. Please set it in Colab secrets or environment variables.")
    
    os.environ["OPENAI_API_KEY"] = api_key
    return api_key

# ===== Helper Functions =====

def clean_byte_literal_str(s: str) -> str:
    """Clean byte literal strings from dataset"""
    if not isinstance(s, str):
        return str(s)
    if s.startswith("b'") and s.endswith("'"):
        s = s[2:-1]
    elif s.startswith('b"') and s.endswith('"'):
        s = s[2:-1]
    s = s.replace('\\n', ' ').replace('\\t', ' ')
    return s.strip()

def is_equivalent(expr1, expr2) -> bool:
    """Check if two mathematical expressions are equivalent using SymPy"""
    try:
        e1 = simplify(sympify(str(expr1), strict=True))
        e2 = simplify(sympify(str(expr2), strict=True))
        eq_result = Eq(e1, e2)
        # Handle both boolean and Eq results
        if hasattr(eq_result, 'equals'):
            return eq_result.equals(True)
        else:
            return bool(eq_result)
    except Exception:
        # Fallback to string comparison
        return str(expr1).strip() == str(expr2).strip()

def detect_problem_type(question: str) -> str:
    """Detect the type of math problem based on keywords"""
    question_lower = question.lower()
    
    if any(keyword in question_lower for keyword in ['factor', 'solve for', 'roots', 'polynomial']):
        return "polynomial"
    elif any(keyword in question_lower for keyword in ['nth term', 'sequence', 'term of']):
        return "sequences"
    elif any(keyword in question_lower for keyword in ['calculate', 'evaluate', 'multiply', 'divide']):
        return "arithmetic"
    elif any(keyword in question_lower for keyword in ['sort', 'increasing', 'decreasing', 'order']):
        return "comparison_sort"
    elif any(keyword in question_lower for keyword in ['base', 'convert', 'binary', 'hexadecimal']):
        return "base_conversion"
    elif any(keyword in question_lower for keyword in ['sqrt', 'simplify', 'surd', 'radical']):
        return "simplify_surd"
    else:
        return "default"

# ===== Main Evaluation Functions =====

def solve_single_question(question: str, problem_type: str = None) -> MathAnswerWithReasoning:
    """
    Solve a single math question using structured output
    
    Args:
        question: The math question to solve
        problem_type: Optional problem type, will auto-detect if None
        
    Returns:
        MathAnswerWithReasoning object with reasoning and final answer
    """
    if problem_type is None:
        problem_type = detect_problem_type(question)
    
    # Setup parser and prompt
    parser = PydanticOutputParser(pydantic_object=MathAnswerWithReasoning)
    format_instructions = parser.get_format_instructions()
    
    system_prompt = PROMPT_TEMPLATES.get(problem_type, PROMPT_TEMPLATES["default"])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{MATH_QUESTION}")
    ]).partial(format_instructions=format_instructions)
    
    # Setup LLM and chain
    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=2)
    chain = prompt | llm | parser
    
    # Get result
    try:
        result = chain.invoke({"MATH_QUESTION": question})
        return result
    except Exception as e:
        print(f"Error processing question: {e}")
        return MathAnswerWithReasoning(
            reasoning=f"Error occurred during processing: {str(e)}",
            final_answer="ERROR"
        )

def run_complete_evaluation(input_filepath: str, output_filepath: str, 
                          problem_type: str = None, max_concurrency: int = 1) -> Dict[str, Any]:
    """
    Run complete evaluation on a dataset file
    
    Args:
        input_filepath: Path to input CSV file
        output_filepath: Path to save results
        problem_type: Optional problem type, will auto-detect if None
        max_concurrency: Max concurrent API calls
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Loading data from {input_filepath}...")
    df = pd.read_csv(input_filepath)
    df['cleaned_question'] = df['question'].apply(clean_byte_literal_str)
    df['ground_truth_cleaned'] = df['answer'].apply(clean_byte_literal_str)
    
    # Auto-detect problem type if not specified
    if problem_type is None:
        sample_question = df['cleaned_question'].iloc[0]
        problem_type = detect_problem_type(sample_question)
        print(f"Auto-detected problem type: {problem_type}")
    
    # Setup chain
    parser = PydanticOutputParser(pydantic_object=MathAnswerWithReasoning)
    format_instructions = parser.get_format_instructions()
    
    system_prompt = PROMPT_TEMPLATES.get(problem_type, PROMPT_TEMPLATES["default"])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{MATH_QUESTION}")
    ]).partial(format_instructions=format_instructions)
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=2)
    chain = prompt | llm | parser
    
    print("Getting answers (with reasoning) from the LLM...")
    questions_to_process = df['cleaned_question'].tolist()
    batch_inputs = [{"MATH_QUESTION": q} for q in questions_to_process]
    
    start_time = time.time()
    try:
        results = chain.batch(batch_inputs, config={"max_concurrency": max_concurrency})
    except Exception as e:
        print(f"Batch processing failed: {e}")
        print("Switching to individual processing...")
        results = []
        for i, inp in enumerate(batch_inputs):
            try:
                result = chain.invoke(inp)
                results.append(result)
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(batch_inputs)} questions")
            except Exception:
                print(f"Skipping question {i+1}")
                results.append(MathAnswerWithReasoning(reasoning="SKIPPED", final_answer="SKIPPED"))
    
    end_time = time.time()
    print(f"LLM processing finished in {end_time - start_time:.2f} seconds.")
    
    # Unpack results
    reasoning_list = []
    llm_answers_list = []
    for res in results:
        if isinstance(res, MathAnswerWithReasoning):
            reasoning_list.append(res.reasoning)
            llm_answers_list.append(res.final_answer)
        else:
            reasoning_list.append("PARSE_ERROR")
            llm_answers_list.append(f"PARSE_ERROR: {res}")
    
    df['reasoning'] = reasoning_list
    df['llm_answer'] = llm_answers_list
    
    print("Comparing LLM answers with ground truth using SymPy...")
    df['is_match'] = df.apply(
        lambda row: is_equivalent(row['llm_answer'], row['ground_truth_cleaned']),
        axis=1
    )
    
    correct_predictions = df['is_match'].sum()
    total_questions = len(df)
    accuracy = (correct_predictions / total_questions * 100) if total_questions > 0 else 0
    
    print(f"\n--- Evaluation Complete ---")
    print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_questions})")
    
    # Save results
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"✅ Detailed comparison results (with reasoning) saved to: {output_filepath}")
    
    return {
        'accuracy': accuracy,
        'correct': correct_predictions,
        'total': total_questions,
        'problem_type': problem_type,
        'processing_time': end_time - start_time
    }

def create_test_subset(original_filepath: str, num_questions: int = 50) -> str:
    """Create a small test subset from a larger dataset"""
    print(f"Reading the full file: {original_filepath}...")
    df_full = pd.read_csv(original_filepath)
    
    df_test_subset = df_full.head(num_questions)
    print(f"Created a small test subset with {len(df_test_subset)} questions.")
    
    temp_test_filepath = original_filepath.replace('.csv', f'_first_{num_questions}.csv')
    df_test_subset.to_csv(temp_test_filepath, index=False)
    print(f"Saved the small test file to: {temp_test_filepath}")
    
    return temp_test_filepath

def quick_test(filepath: str = "arithmetic_simplify_surd.csv", num_questions: int = 5):
    """Quick test function for debugging"""
    print("=== Quick Test ===")
    
    # Setup
    setup_api_key()
    
    # Create test subset
    test_file = create_test_subset(filepath, num_questions)
    
    # Run evaluation
    output_file = f"results/quick_test_{num_questions}.csv"
    results = run_complete_evaluation(test_file, output_file)
    
    print(f"Quick test completed with {results['accuracy']:.1f}% accuracy")
    return results

# ===== Example Usage Functions =====

def run_surd_evaluation(num_questions: int = 50):
    """Run evaluation on arithmetic surd simplification problems"""
    original_filepath = "arithmetic_simplify_surd.csv"
    temp_test_filepath = create_test_subset(original_filepath, num_questions)
    output_file = f"results/surd_evaluation_{num_questions}.csv"
    
    setup_api_key()
    return run_complete_evaluation(temp_test_filepath, output_file, "simplify_surd")

def run_polynomial_evaluation(num_questions: int = 50):
    """Run evaluation on polynomial problems"""
    original_filepath = "algebra_polynomial_roots.csv"
    temp_test_filepath = create_test_subset(original_filepath, num_questions)
    output_file = f"results/polynomial_evaluation_{num_questions}.csv"
    
    setup_api_key()
    return run_complete_evaluation(temp_test_filepath, output_file, "polynomial")

# ===== Main Execution =====

if __name__ == '__main__':
    print("LangChain Math Problem Solver with Chain-of-Thought (CoT) prompting initialized.")
    print("Available functions:")
    print("- quick_test(): Test with 5 questions")
    print("- run_surd_evaluation(): Evaluate surd simplification")
    print("- run_polynomial_evaluation(): Evaluate polynomial problems")
    print("- solve_single_question(): Solve individual questions")
    
    # Example usage
    # quick_test("arithmetic_simplify_surd.csv", 3)