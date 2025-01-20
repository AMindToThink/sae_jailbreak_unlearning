from scripts.editing_models.InterventionModel import InterventionGemmaModel
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
import torch

@register_model("steered")
class InterventionModelLM(HFLM):
    def __init__(self, csv_path, **kwargs):
        self.swap_in_model = InterventionGemmaModel.from_csv(csv_path, device=kwargs.get('device', 'cuda'))
        self.swap_in_model.eval()
        # Initialize other necessary attributes
        super().__init__(pretrained="google/gemma-2-2b", **kwargs)
        if hasattr(self, '_model'):
            # Delete all the model's parameters but keep the object
            for param in self._model.parameters():
                param.data.zero_()
                param.requires_grad = False
            # Remove all model modules while keeping the base object
            for name, module in list(self._model.named_children()):
                delattr(self._model, name)
            torch.cuda.empty_cache()
            
    
    def _model_call(self, inputs):
        # Implement this method to use your model's forward function
        # import pdb;pdb.set_trace()
        return self.swap_in_model.forward(inputs)#