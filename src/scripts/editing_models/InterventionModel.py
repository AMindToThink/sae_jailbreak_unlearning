from .arena_imports import *

from transformer_lens import loading_from_pretrained
from transformers.modeling_outputs import CausalLMOutputWithPast
gemmascope_sae_release = "gemma-scope-2b-pt-res-canonical"
gemmascope_sae_id = "layer_20/width_16k/canonical"

def steering_hook(
    activations: Float[Tensor, "batch pos d_in"],
    hook: HookPoint,
    sae: SAE,
    latent_idx: int,
    steering_coefficient: float,
) -> Tensor:
    """
    Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
    sequence positions.
    """
    return activations + steering_coefficient * sae.W_dec[latent_idx]

class InterventionGemmaModel(HookedSAETransformer):  # Replace with the specific model class
    def __init__(self, fwd_hooks:list, device:str='cuda:0'):
        # config = AutoConfig.from_pretrained("google/gemma-2b")
        base_name = "gemma-2-2b"
        trueconfig = loading_from_pretrained.get_pretrained_model_config(base_name, device=device)
        super().__init__(trueconfig)
        self.model = HookedSAETransformer.from_pretrained(base_name, device=device)
        
        self.fwd_hooks = fwd_hooks
        self.device = device  # Add device attribute
        self.to(device)  # Ensure model is on the correct device
    @classmethod
    def from_csv(cls, csv_path: str, device: str = 'cuda:0') -> 'InterventionGemmaModel':
        """
        Create an InterventionGemmaModel from a CSV file containing steering configurations.
        
        Expected CSV format:
        index, coefficient, sae_id, description
        12082, 240.0, layer_20/width_16k/canonical, increase dogs
        ...

        Args:
            csv_path: Path to the CSV file containing steering configurations
            device: Device to place the model on

        Returns:
            InterventionGemmaModel with configured steering hooks
        """
        import pandas as pd
        
        # Read steering configurations
        df = pd.read_csv(csv_path)
        
        # Create hooks for each row in the CSV
        hooks = []
        for _, row in df.iterrows():
            sae = SAE.from_pretrained(gemmascope_sae_release, row['sae_id'], device=str(device))[0]
            hook = partial(
                steering_hook,
                sae=sae,
                latent_idx=int(row['latent_idx']),
                steering_coefficient=float(row['steering_coefficient'])
            )
            hooks.append((sae.cfg.hook_name, hook))
        
        # Create and return the model
        return cls(fwd_hooks=hooks, device=device)
    def forward(self, *args, **kwargs):
        # Handle both input_ids and direct tensor inputs
        if 'input_ids' in kwargs:
            input_tensor = kwargs.pop('input_ids')  # Use pop to remove it
        elif args:
            input_tensor = args[0]
            args = args[1:]  # Remove the first argument
        else:
            input_tensor = None
    
        with self.model.hooks(fwd_hooks=self.fwd_hooks):
            output = self.model.forward(input_tensor, *args, **kwargs)
        return output

if __name__ == '__main__':
    from transformers import AutoTokenizer
    
    gemma_2_2b_sae = SAE.from_pretrained(gemmascope_sae_release, gemmascope_sae_id, device=str(device))[0]
    # Assuming 'gemma_2_2b' is the model name or path
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    latent_idx = 12082 # represents dogs
    dog_steering_hook = partial(steering_hook, sae=gemma_2_2b_sae, latent_idx=latent_idx, steering_coefficient=240.0)
    my_intervention_model = InterventionGemmaModel(fwd_hooks=[(gemma_2_2b_sae.cfg.hook_name, dog_steering_hook)])
    prompt = "When I look at myself in the mirror, I see"
    print(my_intervention_model.generate(prompt, max_new_tokens=50))

    import argparse
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', action='store_true', help='Save the model')
    parser.add_argument('-p', '--path', type=str, default="intervention_model", help='Path to save the model')
    args = parser.parse_args()

    if args.save:
        save_path = "intervention_model"
        with open(save_path, 'wb') as f:
            pickle.dump(my_intervention_model, f)
        print(f"Model saved to {save_path}")
    # inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # outputs = my_intervention_model(inputs)
    # def decode_output(output_new_forward):
    #     predicted_token_ids = t.argmax(output_new_forward, dim=-1)
    #     print(predicted_token_ids.shape)
    #     print(predicted_token_ids)
    #     # 2. Decode the token IDs back to text
    #     decoded_text = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
        
    #     print(decoded_text)
    #     return decoded_text
    # decode_output(outputs)