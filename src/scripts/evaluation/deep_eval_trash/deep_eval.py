from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Add parent directory to path

from editing_models import InterventionModel
from sae_lens import (
    SAE,
    HookedSAETransformer,
    SAEConfig,
)
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
# base_name = "gemma-2-2b"
from mistral7bdeepbase import get_testable_mistral
from AnyCausalDeepEvalLLM import AnyCausalDeepEvalLLM
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b").to(device)
class TruthyString(str):
    def __bool__(self):
        return True
benchmark = MMLU(tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE], n_shots=3, confinement_instructions=TruthyString(''))
model_to_test = "mistralai/Mistral-7B-v0.1"
results = benchmark.evaluate(model=AnyCausalDeepEvalLLM(model_to_test), batch_size=5)
print("-"*20)
print("Overall Score: ", results)
print("-"*20)
print("Overall Score:", benchmark.overall_score)
print("-"*20)
print("Task-specific Scores: ", benchmark.task_scores)
print("-"*20)
print("Detailed Predictions: ", benchmark.predictions)
print("-"*20)
print(f"Benchmark: {benchmark}")

# Save predictions to CSV
try:
    predictions_df = benchmark.predictions
    safe_model_name = model_to_test.replace('/', '_')
    path_to_save = "../../../results"
    predictions_df.to_csv(f'{path_to_save}/{safe_model_name}_mmlu_predictions.csv', index=False)
except Exception as e:
    print(f"Error saving predictions to CSV: {e}")

import pdb;pdb.set_trace()
pass
pass
pass