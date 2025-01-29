from unlearning.metrics import *
def find_all_permutations_correct(metrics, n_permutations=24):
    """
    Find questions where all permutations were answered correctly.
    
    Parameters:
    - metrics: Dictionary containing metrics from calculate_MCQ_metrics
    - n_permutations: Number of permutations per question (default 24 for all possible permutations)
    
    Returns:
    - correct_question_indices: List of indices where all permutations were correct
    """
    is_correct = metrics['is_correct']
    total_questions = len(is_correct) // n_permutations
    
    # Reshape the is_correct array to be (n_questions, n_permutations)
    correctness_by_question = is_correct.reshape(total_questions, n_permutations)
    
    # Find questions where all permutations are correct
    all_correct_mask = correctness_by_question.all(axis=1)
    correct_question_indices = np.where(all_correct_mask)[0]
    
    return correct_question_indices

def analyze_permutation_performance(metrics, dataset, n_permutations=24):
    """
    Analyze and print detailed performance statistics for permutation results.
    
    Parameters:
    - metrics: Dictionary containing metrics from calculate_MCQ_metrics
    - dataset: The original dataset containing questions
    - n_permutations: Number of permutations per question
    """
    correct_indices = find_all_permutations_correct(metrics, n_permutations)
    
    print(f"Total number of questions: {len(dataset['test'])}")
    print(f"Number of questions with all permutations correct: {len(correct_indices)}")
    print("\nQuestions with all permutations correct:")
    
    for idx in correct_indices:
        # Convert numpy.int64 to Python int
        idx = int(idx)
        question = dataset['test'][idx]['question']
        choices = dataset['test'][idx]['choices']
        print(f"\nQuestion {idx}:")
        print(f"Question: {question}")
        print("Choices:")
        for i, choice in enumerate(choices):
            print(f"{chr(65+i)}. {choice}")
            
    return correct_indices

# To save the results:
def save_correct_indices(indices, output_file):
    """Save the indices of questions with all permutations correct."""
    np.savetxt(output_file, indices, fmt='%d')

from datasets import load_dataset
import numpy as np
import sys
task = sys.argv[2]
# Load the dataset
dataset = load_dataset("cais/wmdp", task)
# Import and load the model from Hugging Face
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(int(sys.argv[3]))

def make_deterministic():
    import random
    import numpy as np
    import torch
    
    # Set seeds
    random.seed(0)  # random_seed
    np.random.seed(1234)  # numpy_seed
    torch.manual_seed(1234)  # torch_seed
    torch.cuda.manual_seed_all(1234)
    
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call this before running the model
make_deterministic()

# Run calculate_MCQ_metrics with all permutations
metrics = calculate_MCQ_metrics(
    model,
    tokenizer=tokenizer,
    dataset_name=task,
    permutations=all_permutations
)
import pdb;pdb.set_trace()
# Find and analyze questions with all permutations correct
correct_indices = analyze_permutation_performance(metrics, dataset)

# Save the indices for future use
save_correct_indices(correct_indices, 'gemma_2_2b_all_perms_correct_deterministic.csv')