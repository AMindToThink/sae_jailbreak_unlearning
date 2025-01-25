import argparse
import subprocess
from tqdm import tqdm
import json
import os
from typing import List

def parse_fewshot_values(fewshot_str: str) -> List[int]:
    """Convert comma-separated string of fewshot values to list of integers."""
    try:
        return [int(x.strip()) for x in fewshot_str.split(',')]
    except ValueError:
        raise ValueError("Few-shot values must be comma-separated integers")

def main():
    # Create parser that accepts all lm-eval arguments plus our custom fewshot argument
    parser = argparse.ArgumentParser(description='Run lm-eval with multiple few-shot values')
    parser.add_argument('--fewshot_values', type=str, default='1',
                        help='Comma-separated list of few-shot values to evaluate')
    
    args, lm_eval_args = parser.parse_known_args()  # Use parse_known_args instead
    
    # Parse few-shot values
    fewshot_values = parse_fewshot_values(args.fewshot_values)
    
    # Run evaluation for each few-shot value
    for num_fewshot in tqdm(fewshot_values, desc='Evaluating few-shot values'):
        # Construct command
        command = ['lm-eval']
        command.extend(lm_eval_args)  # Use the unknown args directly
        command.extend(['--num_fewshot', str(num_fewshot)])
        
        # Run evaluation
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation for {num_fewshot}-shot:")
            print(e.stderr)
            continue

if __name__ == '__main__':
    main()
