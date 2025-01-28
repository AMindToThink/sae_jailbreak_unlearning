# %%
import subprocess
import argparse
from tqdm import tqdm
import torch
import os
# Set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name (e.g., google/gemma-2-2b)")
parser.add_argument("--output_folder", type=str, required=True, help="Base output folder for models and evaluations")
# parser.add_argument("--wmdp_data_dir", type=str, required=True, help="Directory containing WMDP data files")
parser.add_argument("--verbose", action="store_true", help="Enable verbose logging during unlearning")
parser.add_argument("--device", type=str, default='0', help="Device to run on (cuda/cpu)")
# parser.add_argument("--hf_user", type=str, default='AMindToThink', help="The huggingface user who uploaded the file.")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= args.device
# os.environ["WMDP_DATA_DIR"] = args.wmdp_data_dir

# Extract base model name for file naming
base_model_name = args.model_name.split('/')[-1]
assert 'gemma-2' in base_model_name, "The layers here match gemma 2 models in particular. If you want to try something else, you'll have to really understand what you are doing."

# if args.hf_user != '':
#     from huggingface_hub import login
#     login()

# Add torch memory management settings
torch.cuda.empty_cache()
if torch.cuda.is_available():
    # Set memory allocation strategy
    torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of available GPU memory
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

for layer in tqdm([3, 7], desc='layer', position=0):
    for s in tqdm([100, 200, 400], desc=' steering coefficient s', position=1):
        for a in tqdm([100, 300, 500], desc='  alpha', position=2):
            # Clear CUDA cache at the start of each iteration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            layers = f'{layer-2},{layer-1},{layer}'
            saved_model_name = f'{base_model_name}_RMU_s{s}_a{a}_layer{layer}'
            
            # Check if evaluation results already exist
            eval_result_path = f"{args.output_folder}/{saved_model_name}"
            if os.path.exists(eval_result_path):
                print(f"Evaluation results already exist for {saved_model_name}, skipping...")
                continue
            
            # Running the evaluation script
            eval_command = [
                "lm-eval", "--model", "hf",
                "--model_args", f"pretrained={args.hf_user}/{saved_model_name}",
                "--tasks", "mmlu_high_school_us_history,mmlu_high_school_geography,mmlu_human_aging,mmlu_college_computer_science",
                "--batch_size", "8",
                "--output_path", eval_result_path
            ]
            
            result = subprocess.run(eval_command, check=True)
            if result.returncode != 0:
                raise RuntimeError(f"Command failed with return code {result.returncode}")