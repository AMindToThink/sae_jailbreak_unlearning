from lm_eval.models.huggingface import HFLM
from ...editing_models import InterventionGemmaModel
from lm_eval.api.registry import register_model
class HarnessedInterventionModel(HFLM):
    def __init__(self, csv_path, device="cuda"):
        self.model = InterventionGemmaModel.from_csv(csv_path, device=device)
        # Initialize other necessary attributes
        super().__init__(pretrained="dummy", device=device)

    def _model_call(self, inputs):
        # Use the intervention model's forward function
        return self.model.forward(input_ids=inputs)

    # Override other methods as needed, e.g., tokenizer-related methods
from lm_eval.api.registry import register_model

def create_my_custom_lm(intervention_csv_path):
    # Get base path from csv path by removing extension
    model_name = intervention_csv_path.rsplit('.', 1)[0]
    print(f"{model_name=}")
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
