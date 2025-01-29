from datasets import load_dataset
import random
import json
import argparse

def create_short_wmdp(dataset_type, num_questions, all_correct_file=None):
    # Validate dataset type
    assert dataset_type in ['wmdp-bio', 'wmdp-cyber'], "Dataset type must be either 'wmdp-bio' or 'wmdp-cyber'"
    
    # Load the dataset from HuggingFace
    dataset = load_dataset("cais/wmdp", dataset_type)
    train_dataset = dataset['test']
    
    # If all_correct_file is provided, load the indices
    correct_indices = set()
    if all_correct_file:
        with open(all_correct_file, 'r') as f:
            correct_indices = {int(line.strip()) for line in f if line.strip()}
    
    # Collect all valid questions
    questions = []
    for idx, item in enumerate(train_dataset):
        # Skip if all_correct_file is provided and index is not in correct_indices
        if all_correct_file and idx not in correct_indices:
            continue
            
        questions.append({
            'question': item['question'],
            'answer': item['choices'][item['answer']]
        })
    
    # Randomize the order and select specified number of questions
    random.shuffle(questions)
    questions = questions[:num_questions]
    
    # Save to JSON file
    with open(f'wmdp_short_{dataset_type}_{num_questions}.json', 'w') as f:
        for question in questions:
            f.write(json.dumps(question) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_type', type=str, help="Type of dataset to use ('wmdp-bio' or 'wmdp-cyber')")
    parser.add_argument('num_questions', type=int, help="Number of questions to sample")
    parser.add_argument('--all-correct', type=str, help="Path to file containing indices of all correct answers", default=None)
    args = parser.parse_args()
    
    create_short_wmdp(args.dataset_type, args.num_questions, args.all_correct)
