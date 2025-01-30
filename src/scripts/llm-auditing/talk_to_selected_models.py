import pandas as pd
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from InterventionModelCopy2024_01_29 import hooked_from_csv, HookedSAETransformer
from datasets import load_dataset
import argparse
from huggingface_hub import login
import torch
import gc

# Add argument parsing at the top
parser = argparse.ArgumentParser(description='Run model comparisons with HuggingFace authentication')
# parser.add_argument('--hf_token', type=str, required=True, help='HuggingFace API token')
args = parser.parse_args()

# Login to HuggingFace
# login(token=args.hf_token)
# print('logged in')
andres_best = ['/home/cs29824/matthew/sae_jailbreak_unlearning/src/scripts/llm-auditing/selected_steered_csv/top_5_coef_-300_method_none_steer.csv','/home/cs29824/matthew/sae_jailbreak_unlearning/src/scripts/llm-auditing/selected_steered_csv/top_10_coef_-500_method_none_threshold_0.05_const_True_steer.csv'] 

# Load WMPD-Bio dataset from Hugging Face
wmpd_dataset = load_dataset("cais/wmdp", 'wmdp-bio')
print('got dataset')
wmpd_df = pd.DataFrame(wmpd_dataset['test'])  # Convert to DataFrame for easier sampling
print('dataset now dataframe')

# Select 10 random questions
random_questions = wmpd_df['question'].sample(n=5).tolist()
# Initialize models and tokenizer
base_model_name = "google/gemma-2-2b"  # Adjust if using a different base model
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# Function to generate response
def get_model_response(model, prompt, max_new_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda:0')
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    
def ask_model_questions(model, questions, model_name):
    """
    Ask a list of questions to a single model and return the responses
    
    Args:
        model: The loaded model to use
        questions: List of questions to ask
        model_name: Name to display in output
    
    Returns:
        List of responses
    """
    responses = []
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i} to {model_name}: {question}")
        print("-" * 80)
        
        if not isinstance(model, HookedSAETransformer):
            response = get_model_response(model, question)
        else:
            response = model.generate(question, max_new_tokens=20)
            
        print(response)
        print("-" * 40)
        responses.append(response)
    
    return responses

# Create empty DataFrame with desired columns
csv_names = [csv_path.split('/')[-1] for csv_path in andres_best]
columns = ['Question', 'Base'] + csv_names
results_df = pd.DataFrame(columns=columns)

# Load and process base model
print("\nProcessing Base Model (gemma-2-2b):")
base_model = AutoModelForCausalLM.from_pretrained(base_model_name).to('cuda:0')
base_responses = ask_model_questions(base_model, random_questions, base_model_name)
del base_model
clear_gpu_memory()

# Process each steered model
all_steered_responses = []
for csv_path in andres_best:
    csv_name = csv_path.split('/')[-1]
    print(f"\nProcessing Model from {csv_name}:")
    model = hooked_from_csv(csv_path, base_model_name).to('cuda:0')
    responses = ask_model_questions(model, random_questions, f"Steered Model ({csv_name})")
    all_steered_responses.append(responses)
    del model
    clear_gpu_memory()

# Build DataFrame
for i, question in enumerate(random_questions):
    row_data = {
        'Question': question,
        'Base': base_responses[i]
    }
    for csv_name, responses in zip(csv_names, all_steered_responses):
        row_data[csv_name] = responses[i]
    results_df.loc[len(results_df)] = row_data

# Save or display the DataFrame at the end
print("\nFinal Results DataFrame:")
print(results_df)
save_to_name = 'does_refusal_steer_cause_refusal.csv'
results_df.to_csv(save_to_name, index=False)
print(f"\nResults saved to '{save_to_name}'")

