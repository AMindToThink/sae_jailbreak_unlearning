from scripts.editing_models.InterventionModel import InterventionGemmaModel
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model

from transformers import AutoModelForCausalLM
class HarnessedInterventionModel(HFLM):
    def __init__(self, csv_path, device="cuda"):
        self.swap_in_model = InterventionGemmaModel.from_csv(csv_path, device=device)
        # Initialize other necessary attributes
        super().__init__(pretrained=AutoModelForCausalLM.from_pretrained("google/gemma-2b"), device=device)
    
    def _model_call(self, inputs):
        # Implement this method to use your model's forward function
        return self.swap_in_model.forward(inputs)
    # Override other methods as needed, e.g., tokenizer-related methods
from lm_eval.api.registry import register_model

def create_my_custom_lm(intervention_csv_path):
    # Get filename by taking last part of path and removing extension
    model_name = intervention_csv_path.split('/')[-1].rsplit('.', 1)[0]
    model_name = intervention_csv_path.rsplit('.', 1)[0]
    print(f"Registering {model_name=}")
    @register_model(model_name)
    class WrappedCustomLM(HarnessedInterventionModel):
        def __init__(self):
            super().__init__(intervention_csv_path)
    
    return WrappedCustomLM()
if __name__ == '__main__':
    import sys
    # 
    intervention_csv_path = sys.argv[1]
    model = create_my_custom_lm(intervention_csv_path)
