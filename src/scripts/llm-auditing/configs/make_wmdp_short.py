from datasets import load_dataset
import random
import json

def create_short_wmdp():
    # Load the dataset from HuggingFace
    dataset = load_dataset("cais/wmdp", 'wmdp-bio')
    # Access the 'train' split (or whichever split you want to use)
    train_dataset = dataset['test']
    int_to_str_answer = ['A', 'B',"C","D"]
    # Create a dictionary to store questions by their answer
    questions_by_answer = {0: [], 1: [], 2: [], 3: []}
    # import pdb;pdb.set_trace()
    # Sort questions by their answer
    for item in train_dataset:
        answer = item['answer']
        question = item['question']
        choices = item['choices']
        
        # Format question with lettered choices
        formatted_question = f"{question}\n"
        for i, choice in enumerate(choices):
            formatted_question += f"{chr(65+i)}. {choice}\n"
        formatted_question += '\nAnswer: '
        if answer in questions_by_answer:
            questions_by_answer[answer].append({
                'question': formatted_question.strip(),  # strip to remove trailing newline
                'answer': int_to_str_answer[answer]
            })
    
    # Sample 4 questions from each answer category
    sampled_questions = []
    for answer in [0,1,2,3]:
        if questions_by_answer[answer]:
            samples = random.sample(questions_by_answer[answer], min(4, len(questions_by_answer[answer])))
            sampled_questions.extend(samples)
    
    # Randomize the order of all sampled questions
    random.shuffle(sampled_questions)
    
    # Save to JSON file
    with open('wmdp_short.json', 'w') as f:
        for question in sampled_questions:
            f.write(json.dumps(question) + '\n')

if __name__ == "__main__":
    create_short_wmdp()
