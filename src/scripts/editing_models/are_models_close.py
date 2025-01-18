# usage from src: PYTHONPATH=. python -m scripts.editing_models.are_models_close --use_hf --csv1 ../models/dog_steer.csv --device cuda:1
raise NotImplemented
from .InterventionModel import InterventionGemmaModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_hf', action='store_true', help='Use HuggingFace model as first model')
    parser.add_argument('--csv1', type=str, required=True, help='Path to first steering CSV (or only CSV if using HF model)')
    parser.add_argument('--csv2', type=str, help='Path to second steering CSV (required if not using HF model)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run models on (cuda/cpu)')
    args = parser.parse_args()

    if not args.use_hf and args.csv2 is None:
        parser.error("--csv2 is required when not using HuggingFace model")

    # Create models and evaluate one at a time
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    prompt = "When I look at myself in the mirror, I see"
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(args.device)
    
    # Get outputs from first model
    if args.use_hf:
        model1 = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b").to(args.device)
        with torch.no_grad():
            outputs1 = model1.forward(inputs).logits
        del model1  # Free up GPU memory
    else:
        model1 = InterventionGemmaModel.from_csv(args.csv1)
        with torch.no_grad():
            outputs1 = model1(inputs)
        del model1

    # Get outputs from second model
    model2 = InterventionGemmaModel.from_csv(args.csv1 if args.use_hf else args.csv2)
    with torch.no_grad():
        outputs2 = model2(inputs)
    
    # Compare logits
    are_close = torch.allclose(outputs1, outputs2, rtol=1e-4, atol=1e-4)
    print(f"Outputs are {'the same' if are_close else 'different'}")
    if not are_close:
        diff = (outputs1 - outputs2).abs().mean()
        print(f"Average absolute difference: {diff:.6f}")