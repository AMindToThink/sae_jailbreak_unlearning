from datasets import load_dataset, DatasetDict
import argparse
from huggingface_hub import login
from datasets.features import Value

def load_and_combine_datasets(username):
    # Load MMLU college_biology dataset
    mmlu = load_dataset("cais/mmlu", "college_biology")
    
    # Remove 'subject' column from MMLU dataset
    mmlu_cleaned = mmlu.remove_columns(['subject'])
    
    
    
    mmlu_converted = mmlu_cleaned.cast_column('answer', Value(dtype='int64'))
    
    # Load WMDP dataset and filter for bio subset
    wmdp = load_dataset("cais/wmdp", "wmdp-bio")
    
    # Print dataset features to debug
    print("MMLU features:", mmlu_converted['validation'].features)
    print("WMDP features:", wmdp['test'].features)
    
    # Create a new DatasetDict with MMLU as validation and WMDP as test
    combined_dataset = DatasetDict({
        'mmlu_test':mmlu_converted['test'],
        'mmlu_validation': mmlu_converted['validation'],
        'wmdp_test': wmdp['test']
    })
    
    # Save locally and push to hub
    dataset_name = f"{username}/mmlu_wmdp_bio_combined"
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
    print(f"Validation set (MMLU biology) size: {len(combined_dataset['mmlu_validation'])}")
    print(f"Test set (WMDP bio) size: {len(combined_dataset['wmdp_test'])}")
