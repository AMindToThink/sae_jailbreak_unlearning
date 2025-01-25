from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import sys

# Log in to Hugging Face Hub
login()

# Parse arguments
arg_counter = 1
username = sys.argv[arg_counter]
arg_counter += 1
path = sys.argv[arg_counter]

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

# Extract model name from path
model_name = path.split('/')[-1]
readme_str = """This is a gemma-2-2b-it model trained using RMU to be worse at answering questions about bioweapons. s is the unlearning coefficient aka 'c', and a is the weight of the retain loss.
                    https://arxiv.org/pdf/2403.03218 
                    This is part of a series to replicate and extend the discoveries [here](https://arxiv.org/abs/2410.19278).
                    The code that literally actually made this model is [here](https://github.com/AMindToThink/wmdp)."""
                    
# Push model to hub
model.push_to_hub(f"{username}/{model_name}", readme_file=readme_str)

# Push tokenizer to hub
tokenizer.push_to_hub(f"{username}/{model_name}")

print(f"Model and tokenizer have been uploaded to Hugging Face Hub under '{username}/{model_name}'!")