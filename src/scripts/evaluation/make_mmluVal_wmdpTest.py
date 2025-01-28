from datasets import load_dataset, DatasetDict, Dataset
import argparse
from huggingface_hub import login
from datasets.features import Value

def load_and_combine_datasets(username):
    # Load MMLU datasets for different subjects
    mmlu_subjects = {
        'bio': 'college_biology',
        'cs': 'college_computer_science',
        'chem': 'college_chemistry',
    }
    
    # Additional MMLU subjects for the combined split
    extra_mmlu_subjects = [
        'high_school_us_history',
        'high_school_geography',
        'human_aging',
        'college_computer_science'
    ]
    
    mmlu_datasets = {}
    # mmlu_test_combined = []
    extra_combined = []
    
    # Load main MMLU subjects
    for key, subject in mmlu_subjects.items():
        dataset = load_dataset("cais/mmlu", subject)
        cleaned = dataset.remove_columns(['subject'])
        converted = cleaned.cast_column('answer', Value(dtype='int64'))
        mmlu_datasets[key] = converted
        # mmlu_test_combined.extend(converted['test'])
    
    # Load and combine extra MMLU subjects
    for subject in extra_mmlu_subjects:
        dataset = load_dataset("cais/mmlu", subject)
        cleaned = dataset.remove_columns(['subject'])
        converted = cleaned.cast_column('answer', Value(dtype='int64'))
        extra_combined.extend(converted['test'])
    
    # Load WMDP datasets for different classes
    wmdp_classes = ['wmdp-bio', 'wmdp-cyber', 'wmdp-chem']
    wmdp_datasets = {
        class_name: load_dataset("cais/wmdp", class_name)['test']
        for class_name in wmdp_classes
    }
    
    # Convert extra_combined to a Dataset
    extra_combined_dataset = Dataset.from_list(extra_combined)
    
    # Create a new DatasetDict with all splits
    combined_dataset = DatasetDict({
        'mmlu_biology': mmlu_datasets['bio']['test'],
        'mmlu_computer_science': mmlu_datasets['cs']['test'],
        'mmlu_chemistry': mmlu_datasets['chem']['test'],
        'mmlu_mmlu_high_school_us_history_and_mmlu_high_school_geography_and_mmlu_human_aging_and_mmlu_college_computer_science_combined': extra_combined_dataset,  # New combined split
        'wmdp_bio': wmdp_datasets['wmdp-bio'],
        'wmdp_cyber': wmdp_datasets['wmdp-cyber'],
        'wmdp_chem': wmdp_datasets['wmdp-chem']
    })
    
    # Save locally and push to hub
    dataset_name = f"{username}/mmlu_wmdp_combined"
    combined_dataset.push_to_hub(dataset_name, private=True)
    
    return combined_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create and upload combined MMLU-WMDP dataset')
    parser.add_argument('--token', required=True, help='HuggingFace API token')
    parser.add_argument('--username', required=True, help='HuggingFace username')
    args = parser.parse_args()

    # Login to Hugging Face
    login(token=args.token)
    
    print("Creating combined dataset...")
    combined_dataset = load_and_combine_datasets(args.username)
    print("Dataset created and uploaded successfully!")
    print(f"MMLU biology validation set size: {len(combined_dataset['mmlu_biology'])}")
    print(f"MMLU computer science validation set size: {len(combined_dataset['mmlu_computer_science'])}")
    print(f"MMLU chemistry validation set size: {len(combined_dataset['mmlu_chemistry'])}")
    print(f"WMDP bio test set size: {len(combined_dataset['wmdp_bio'])}")
    print(f"WMDP cyber test set size: {len(combined_dataset['wmdp_cyber'])}")
    print(f"WMDP chem test set size: {len(combined_dataset['wmdp_chem'])}")
