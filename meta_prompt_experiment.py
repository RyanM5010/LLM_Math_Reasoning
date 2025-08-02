import pandas as pd
from openai import OpenAI
import time
import os
from google.colab import userdata

# Get API key from Colab secret
api_key = userdata.get('openai_api')
if api_key is None:
    raise ValueError("Missing secret: openai_api")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# ===== Experiment A: Over-constraint Hypothesis Testing =====

# A1: Simplest prompt (baseline)
SIMPLE_PROMPT = """Solve: {{MATH_QUESTION}}"""

# A2: Original Meta_Prompt (control group)
META_PROMPT = """You are a mathematical problem solver. You will be given a single math question to solve. Your task is to return only the final answer, without showing any intermediate steps, explanations, or reasoning.

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

# ===== Experiment B: Format Anxiety Hypothesis Testing =====

# B1: Remove all formatting requirements
NO_FORMAT_PROMPT = """You are a mathematical problem solver.
Solve this problem and give me the answer: {{MATH_QUESTION}}"""

# B2: Simplified formatting requirements
SIMPLE_FORMAT_PROMPT = """Solve: {{MATH_QUESTION}}
Just give the final answer, nothing else."""

# ===== Experiment C: Internal Reasoning Suppression Hypothesis Testing =====

# C1: Completely prohibit reasoning
NO_REASONING_PROMPT = """Solve internally. Show no work. Only return final answer: {{MATH_QUESTION}}"""

# C2: Allow reasoning but require brevity
BRIEF_REASONING_PROMPT = """Solve this briefly, then give final answer:
{{MATH_QUESTION}}"""

# C3: Allow full reasoning
FULL_REASONING_PROMPT = """Think step by step to solve:
{{MATH_QUESTION}}
Final answer:"""

# Dictionary of all prompt variants
PROMPT_VARIANTS = {
    # Experiment A
    "A1_Simple": SIMPLE_PROMPT,
    "A2_Meta": META_PROMPT,

    # Experiment B
    "B1_NoFormat": NO_FORMAT_PROMPT,
    "B2_SimpleFormat": SIMPLE_FORMAT_PROMPT,
    "B3_Meta": META_PROMPT,  # Repeated as control

    # Experiment C
    "C1_NoReasoning": NO_REASONING_PROMPT,
    "C2_BriefReasoning": BRIEF_REASONING_PROMPT,
    "C3_FullReasoning": FULL_REASONING_PROMPT
}

def get_answer_with_prompt(question, prompt_template, retries=3, delay=2):
    """Get answer using specified prompt template"""
    for attempt in range(retries):
        try:
            # Replace placeholder with actual question
            if "{{MATH_QUESTION}}" in prompt_template:
                prompt = prompt_template.replace("{{MATH_QUESTION}}", question)
                messages = [{"role": "system", "content": prompt}]
            else:
                # For simple prompts, use directly as user message
                messages = [{"role": "user", "content": prompt_template.replace("{{MATH_QUESTION}}", question)}]

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=1500  # Increased token limit for potential reasoning output
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                print(f"Retrying ({attempt + 1}/{retries}) after error: {e}")
                time.sleep(delay)
            else:
                return f"ERROR: {str(e)}"

def run_experiment(file_name, experiment_type="A", input_dir="./", output_dir="./experiment_outputs",
                  save_every=5, sleep_time=1, max_questions=50):
    """
    Run specified experiment
    experiment_type: "A", "B", "C", or "ALL"
    max_questions: Maximum number of test questions (default 50)
    """
    os.makedirs(output_dir, exist_ok=True)

    # If file_name already contains path, use directly; otherwise join with input_dir
    if "/" in file_name:
        input_path = file_name
        file_base_name = os.path.basename(file_name)
    else:
        input_path = os.path.join(input_dir, file_name)
        file_base_name = file_name

    # Select prompt variants based on experiment type
    if experiment_type == "A":
        prompts_to_test = {k: v for k, v in PROMPT_VARIANTS.items() if k.startswith("A")}
    elif experiment_type == "B":
        prompts_to_test = {k: v for k, v in PROMPT_VARIANTS.items() if k.startswith("B")}
    elif experiment_type == "C":
        prompts_to_test = {k: v for k, v in PROMPT_VARIANTS.items() if k.startswith("C")}
    elif experiment_type == "ALL":
        prompts_to_test = PROMPT_VARIANTS
    else:
        raise ValueError("experiment_type must be 'A', 'B', 'C', or 'ALL'")

    df = pd.read_csv(input_path)

    # Only take first max_questions
    original_length = len(df)
    df = df.head(max_questions)
    print(f"Original data: {original_length} questions, testing first {len(df)} questions")

    # Create result columns for each prompt variant
    for prompt_name in prompts_to_test.keys():
        df[f'{prompt_name}_answer'] = ""
        df[f'{prompt_name}_length'] = 0

    total_questions = len(df)

    for idx, row in df.iterrows():
        question = row['question']
        print(f"\n[{file_base_name}] Processing {idx + 1}/{total_questions}: {question[:50]}...")

        # Test each prompt variant
        for prompt_name, prompt_template in prompts_to_test.items():
            print(f"  Testing {prompt_name}...")
            answer = get_answer_with_prompt(question, prompt_template)
            df.loc[idx, f'{prompt_name}_answer'] = answer
            df.loc[idx, f'{prompt_name}_length'] = len(answer)
            time.sleep(sleep_time)

        # Save progress periodically
        if (idx + 1) % save_every == 0:
            output_path = os.path.join(output_dir, f"{file_base_name.replace('.csv', '')}_experiment_{experiment_type}_top{max_questions}.csv")
            df.to_csv(output_path, index=False)
            print(f"Progress saved: {idx + 1} questions")

    # Final save
    output_path = os.path.join(output_dir, f"{file_base_name.replace('.csv', '')}_experiment_{experiment_type}_top{max_questions}.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Experiment {experiment_type} completed: {file_base_name} (first {max_questions} questions)")

    return df

def analyze_experiment_results(df, experiment_type="A"):
    """Analyze experiment results"""
    if experiment_type == "A":
        prompt_names = ["A1_Simple", "A2_Meta"]
        print("\n=== Experiment A: Over-constraint Hypothesis Analysis ===")
    elif experiment_type == "B":
        prompt_names = ["B1_NoFormat", "B2_SimpleFormat", "B3_Meta"]
        print("\n=== Experiment B: Format Anxiety Hypothesis Analysis ===")
    elif experiment_type == "C":
        prompt_names = ["C1_NoReasoning", "C2_BriefReasoning", "C3_FullReasoning"]
        print("\n=== Experiment C: Internal Reasoning Suppression Hypothesis Analysis ===")

    # Analyze response length (proxy for reasoning complexity)
    print("\nResponse length statistics:")
    for prompt_name in prompt_names:
        if f'{prompt_name}_length' in df.columns:
            avg_length = df[f'{prompt_name}_length'].mean()
            print(f"{prompt_name}: Average length = {avg_length:.1f} characters")

    # Show some example answers for qualitative analysis
    print(f"\nFirst 3 questions' response examples:")
    for i in range(min(3, len(df))):
        print(f"\nQuestion {i+1}: {df.iloc[i]['question'][:60]}...")
        for prompt_name in prompt_names:
            if f'{prompt_name}_answer' in df.columns:
                answer = df.iloc[i][f'{prompt_name}_answer']
                print(f"  {prompt_name}: {answer}")

# ===== Usage Examples =====

def quick_test_single_file(file_name="algebra__polynomial_roots.csv"):
    """Quick test on single file, only run first 3 questions"""
    if not os.path.exists(file_name):
        print(f"File does not exist: {file_name}")
        print("Please check file path and name")
        return None

    df = pd.read_csv(file_name)
    print(f"File loaded successfully: {file_name}")
    print(f"Total {len(df)} questions")
    print(f"First 3 questions:")
    for i in range(min(3, len(df))):
        print(f"  {i+1}: {df.iloc[i]['question']}")

    # Only test first 3 questions for experiment A
    df_small = df.head(3).copy()

    prompts_to_test = {
        "A1_Simple": SIMPLE_PROMPT,
        "A2_Meta": META_PROMPT
    }

    for prompt_name, prompt_template in prompts_to_test.items():
        answers = []
        for idx, row in df_small.iterrows():
            question = row['question']
            print(f"Testing {prompt_name} on Q{idx+1}...")
            answer = get_answer_with_prompt(question, prompt_template)
            answers.append(answer)
            time.sleep(1)
        df_small[f'{prompt_name}_answer'] = answers

    print("\n=== Quick Test Results ===")
    for i in range(len(df_small)):
        print(f"\nQ{i+1}: {df_small.iloc[i]['question']}")
        print(f"  Simple: {df_small.iloc[i]['A1_Simple_answer']}")
        print(f"  Meta:   {df_small.iloc[i]['A2_Meta_answer']}")

    return df_small

# Run single experiment
# df_exp_a = run_experiment("your_file.csv", experiment_type="A")
# analyze_experiment_results(df_exp_a, "A")

# Run all experiments
# df_all = run_experiment("algebra_polynomial_roots", experiment_type="ALL")

def batch_experiment(file_list, experiment_type="A", max_questions=50):
    """Batch process multiple files"""
    results = {}
    for file_name in file_list:
        print(f"\n{'='*50}")
        print(f"Processing {os.path.basename(file_name)} (first {max_questions} questions)")
        print('='*50)
        results[file_name] = run_experiment(file_name, experiment_type, max_questions=max_questions)
        time.sleep(5)  # Rest between files
    return results

# ===== File Path Check and Fix =====

def check_files():
    """Check if files exist and show correct paths"""
    import glob

    print("Checking CSV files in current directory and subdirectories:")

    # Check different possible locations
    possible_locations = [
        "/content/*.csv",
        "/content/sample_data/*.csv",
        "./sample_data/*.csv",
        "./*.csv"
    ]

    all_csv_files = []
    for pattern in possible_locations:
        files = glob.glob(pattern)
        if files:
            print(f"\nFound in {pattern}:")
            for f in files:
                print(f"  {f}")
                all_csv_files.append(f)

    # Look for our 6 target topic files - corrected to single underscore
    target_topics = [
        "algebra_polynomial_roots",
        # "algebra_sequence_nth_term",
        # "arithmetic_mul_div_multiple",
        # "arithmetic_simplify_surd",
        # "comparison_sort_composed",
        # "numbers_base_conversion"
    ]

    found_files = []
    print(f"\nLooking for target files:")
    for topic in target_topics:
        matches = [f for f in all_csv_files if topic in f]
        if matches:
            print(f"  ✅ {topic}: {matches[0]}")
            found_files.append(matches[0])
        else:
            print(f"  ❌ {topic}: Not found")

    return found_files

# Run file check
print("Checking file locations...")
available_files = check_files()

# Update file list to actually found files
if available_files:
    topics_files = available_files
    print(f"\nWill use the following {len(topics_files)} files:")
    for f in topics_files:
        print(f"  {f}")
else:
    print("\nTarget files not found, please check if files are uploaded correctly")
    topics_files = []

# Run experiment A
results_a = batch_experiment(topics_files, "A")

# Run experiment B
results_b = batch_experiment(topics_files, "B")

# Run experiment C
results_c = batch_experiment(topics_files, "C")