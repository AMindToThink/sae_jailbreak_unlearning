from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Add parent directory to path

from mistral7bdeepbase import get_testable_mistral
from editing_models import InterventionModel
from sae_lens import (
    SAE,
    HookedSAETransformer,
    SAEConfig,
)
import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
# base_name = "gemma-2-2b"
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b").to(device)
benchmark = MMLU(tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY])
results = benchmark.evaluate(model=get_testable_mistral(), batch_size=5)
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