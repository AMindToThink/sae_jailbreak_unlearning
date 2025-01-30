# This is a copy of Intervention Model as it was on 2024-01-29 for use in our adversarial evaluation. 

# Really, this script should be published to SAELens, not Eleuther's Evaluation Harness
from functools import partial

import torch
from jaxtyping import Float
from sae_lens import SAE, HookedSAETransformer
from torch import Tensor
from transformer_lens import loading_from_pretrained
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer

def steering_hook_add_scaled_one_hot(
    activations,#: Float[Tensor],  # Float[Tensor, "batch pos d_in"], Either jaxtyping or lm-evaluation-harness' precommit git script hate a type hint here.
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

# def steering_hook_clamp(
#     activations,#: Float[Tensor],  # Float[Tensor, "batch pos d_in"], Either jaxtyping or lm-evaluation-harness' precommit git script hate a type hint here.
#     hook: HookPoint,
#     sae: SAE,
#     latent_idx: int,
#     steering_coefficient: float,
# ) -> Tensor:
#     """
#     Steers the model by returning a modified activations tensor, with some multiple of the steering vector added to all
#     sequence positions.
#     """
#     raise NotImplemented
#     z = sae.encode(activations)
#     z[latent_idx] = steering_coefficient
#     return sae.decode(activations)
#     return activations + steering_coefficient * sae.W_dec[latent_idx]


def clamp_sae_feature(sae_acts:Tensor, hook:HookPoint, latent_idx:int, value:float) -> Tensor:
    """Clamps a specific latent feature in the SAE activations to a fixed value.

    Args:
        sae_acts (Tensor): The SAE activations tensor, shape [batch, pos, features]
        hook (HookPoint): The transformer-lens hook point
        latent_idx (int): Index of the latent feature to clamp
        value (float): Value to clamp the feature to

    Returns:
        Tensor: The modified SAE activations with the specified feature clamped
    """
    
    sae_acts[:, :, latent_idx] = value

    return sae_acts

def clamp_original(sae_acts:Tensor, hook:HookPoint, latent_idx:int, value:float) -> Tensor:
    """Clamps a specific latent feature in the SAE activations to a fixed value.

    Args:
        sae_acts (Tensor): The SAE activations tensor, shape [batch, pos, features]
        hook (HookPoint): The transformer-lens hook point
        latent_idx (int): Index of the latent feature to clamp
        value (float): Value to clamp the feature to

    Returns:
        Tensor: The modified SAE activations with the specified feature clamped
    """
    #import pdb;pdb.set_trace()
    mask = sae_acts[:, :, latent_idx] > 0  # Create a boolean mask where values are greater than 0
    sae_acts[:, :, latent_idx][mask] = value  # Replace values conditionally

    return sae_acts

def clamp_conditional(sae_acts:Tensor, hook:HookPoint, latent_idx:int, value:float, clamp_value:float) -> Tensor:
    """Clamps a specific latent feature in the SAE activations to a fixed value over a threshold.

    Args:
        sae_acts (Tensor): The SAE activations tensor, shape [batch, pos, features]
        hook (HookPoint): The transformer-lens hook point
        latent_idx (int): Index of the latent feature to clamp
        value (float): Value to clamp the feature to

    Returns:
        Tensor: The modified SAE activations with the specified feature clamped
    """
    mask = sae_acts[:, :, latent_idx] > clamp_value  # Create a boolean mask where values are greater than 0
    sae_acts[:, :, latent_idx][mask] = value  # Replace values conditionally
    return sae_acts

def clamp_refusal(sae_acts:Tensor, hook:HookPoint, latent_idx:int, value:float, clamp_value:float, refusal_idx:int) -> Tensor:
    """Clamps a specific latent feature in the SAE activations to a fixed value.

    Args:
        sae_acts (Tensor): The SAE activations tensor, shape [batch, pos, features]
        hook (HookPoint): The transformer-lens hook point
        latent_idx (int): Index of the latent feature to clamp
        value (float): Value to clamp the feature to
        refusal_idx (int): Index of the latent that is related to the refusal

    Returns:
        Tensor: The modified SAE activations with the specified feature clamped
    """
    REFUSAL = refusal_idx
    mask = sae_acts[:, :, latent_idx] > clamp_value  # Create a boolean mask where values are greater than 0
    sae_acts[:, :, REFUSAL][mask] = value  # Replace values conditionally

    return sae_acts

def print_sae_acts(sae_acts:Tensor, hook:HookPoint) -> Tensor:
    """Clamps a specific latent feature in the SAE activations to a fixed value.

    Args:
        sae_acts (Tensor): The SAE activations tensor, shape [batch, pos, features]
        hook (HookPoint): The transformer-lens hook point
        latent_idx (int): Index of the latent feature to clamp
        value (float): Value to clamp the feature to

    Returns:
        Tensor: The modified SAE activations with the specified feature clamped
    """
    print(40*"----")
    print(f"This is the latent activations of {hook.name}")
    print(sae_acts.shape)
    print(torch.all(sae_acts > 0))
    return sae_acts
def debug_steer(sae_acts: Tensor, hook:HookPoint) -> Tensor:
    import pdb; pdb.set_trace()
    pass
    pass
    return sae_acts

string_to_steering_function_dict : dict = {'add':steering_hook_add_scaled_one_hot, 
                                           'clamp':clamp_original,
                                           'clamp_cond':clamp_conditional,
                                           'print':print_sae_acts,
                                           'debug':debug_steer,
                                           }
def hooked_from_dataframe(df, base_name:str, device:str='cuda:0', dtype=torch.float32):
    model = HookedSAETransformer.from_pretrained_no_processing(base_name, device=device, dtype=dtype)
    original_saes = model.acts_to_saes
    assert original_saes == {} # There shouldn't be any SAEs to start
    # Read steering configurations
    # Create hooks for each row in the CSV
    sae_cache = {}
    # original_sae_hooks_cache = {}
    def get_sae(sae_release, sae_id):
        cache_key = (sae_release, sae_id)
        if cache_key not in sae_cache:
            sae_cache[cache_key] = SAE.from_pretrained(
                sae_release, sae_id, device=str(device)
            )[0]
            # original_sae_hooks_cache[cache_key] = sae_cache[cache_key]
        return sae_cache[cache_key]

    for _, row in df.iterrows():
        sae_release = row["sae_release"]
        sae_id = row["sae_id"]
        latent_idx = int(row["latent_idx"])
        steering_coefficient = float(row["steering_coefficient"])
        clamp_value = float(row['clamp_value']) if 'clamp_value' in row else None
        refusal_idx = int(row['refuse_id']) if 'refuse_id' in row else None

        sae = get_sae(sae_release=sae_release, sae_id=sae_id)
        sae.use_error_term = True
        sae.eval()
        # Add the SAE to the model after configuring its hooks
        model.add_sae(sae)
        # First add all hooks to the SAE before adding it to the model
        hook_action = row.get("hook_action", "add")
        after_activation_fn = f"{sae.cfg.hook_name}.hook_sae_acts_post"
        if hook_action == "add":
            hook_name = f"{sae.cfg.hook_name}.hook_sae_input" # we aren't actually putting the input through the model
            hook = partial(steering_hook_add_scaled_one_hot,
                            sae=sae,
                            latent_idx=latent_idx,
                            steering_coefficient=steering_coefficient,
                            )
            model.add_hook(hook_name, hook)
        elif hook_action == "clamp":
            sae.add_hook("hook_sae_acts_post", partial(clamp_original, latent_idx=latent_idx, value=steering_coefficient))
        elif hook_action == "clamp_cond":
            sae.add_hook("hook_sae_acts_post", partial(clamp_conditional, latent_idx=latent_idx, value=steering_coefficient, clamp_value=clamp_value))
        elif hook_action == "clamp_refusal":
            sae.add_hook("hook_sae_acts_post", partial(clamp_refusal, latent_idx=latent_idx, value=steering_coefficient, clamp_value=clamp_value, refusal_idx=refusal_idx))
        elif hook_action == 'print':
            model.add_hook(after_activation_fn, print_sae_acts)
        elif hook_action == 'debug':
            model.add_hook(after_activation_fn, debug_steer)
        else:
            raise ValueError(f"Unknown hook type: {hook_action}")
        
        
    # Create and return the model
    return model

def hooked_from_csv(
    csv_path: str, base_name: str, device: str = "cuda:0", dtype=torch.float32
) -> HookedSAETransformer:
    """
    Create an InterventionModel from a CSV file containing steering configurations.

    Expected CSV format:
    index, coefficient, sae_release, sae_id, description
    12082, 240.0,gemma-scope-2b-pt-res-canonical,layer_20/width_16k/canonical, increase dogs
    ...

    Args:
        csv_path: Path to the CSV file containing steering configurations
        device: Device to place the model on

    Returns:
        InterventionModel with configured steering hooks
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    return hooked_from_dataframe(df=df, base_name=base_name, device=device, dtype=dtype)

class InterventionModel(HookedSAETransformer):  # Replace with the specific model class
    def __init__(self, base_name: str, device: str = "cuda:0", model=None, dtype=torch.float32):
        trueconfig = loading_from_pretrained.get_pretrained_model_config(
            base_name, device=device, dtype=dtype
        )
        super().__init__(trueconfig)
        self.model = model or HookedSAETransformer.from_pretrained_no_processing(base_name, device=device, dtype=dtype)
        self.model.use_error_term = True
        self.model.eval()
        self.device = device  # Add device attribute
        self.to(device)  # Ensure model is on the correct device
        
        # Replace with nn.Embedding
        num_embeddings, embedding_dim = self.model.W_E.shape
        embed_layer = torch.nn.Embedding(num_embeddings, embedding_dim)
        embed_layer.weight = torch.nn.Parameter(self.model.W_E)
        self.model.embed_tokens = embed_layer

    @classmethod
    def from_dataframe(cls, df, base_name:str, device:str='cuda:0', dtype=torch.float32):
        model = hooked_from_dataframe(df=df, base_name=base_name, device=device, dtype=dtype)
        return cls(base_name=base_name, device=device, model=model, dtype=dtype)

    @classmethod
    def from_csv(cls, csv_path: str, base_name: str, device: str = "cuda:0", dtype=torch.float32) -> "InterventionModel":
        model = hooked_from_csv(csv_path=csv_path, base_name=base_name, device=device, dtype=dtype)
        return cls(base_name=base_name, device=device, model=model, dtype=dtype)

    def forward(self, *args, **kwargs):
        # Handle both input_ids and direct tensor inputs
        if "input_ids" in kwargs:
            input_tensor = kwargs.pop("input_ids")
            kwargs["input"] = input_tensor
        elif "inputs_embeds" in kwargs:
            input_embeds = kwargs.pop("inputs_embeds")
            kwargs["input"] = input_embeds
            kwargs["start_at_layer"] = 0
        elif args:
            input_tensor = args[0]
            args = args[1:]
            kwargs["input"] = input_tensor
        else:
            raise ValueError("No input provided to forward pass")
            
        with torch.no_grad():# I don't know why this no grad is necessary; I tried putting everything into eval mode. And yet, this is necessary to prevent CUDA out of memory exceptions.
            output = self.model.forward(*args, **kwargs)
        class OutputWithLogits:
            def __init__(self, logits):
                self.logits = logits
    
        return OutputWithLogits(output)

    def generate(self, *args, **kwargs):
        # Handle both input_ids and direct tensor inputs
        if "input_ids" in kwargs:
            input_tensor = kwargs.pop("input_ids")  # Use pop to remove it
        elif args:
            input_tensor = args[0]
            args = args[1:]  # Remove the first argument
        else:
            input_tensor = None
        with torch.no_grad():  # I don't know why this no grad is necessary; I tried putting everything into eval mode. And yet, this is necessary to prevent CUDA out of memory exceptions.
            output = self.model.generate(input_tensor, *args, **kwargs)
        return output

if __name__ == '__main__':
    # Initialize the model and tokenizer
    model_name = 'google/gemma-2-2b'
    model = InterventionModel(model_name, device="cuda:0")  # or whatever model you're using
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create input text and tokenize
    text = "Hello, world!"
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda:0")

    # First, get the embeddings from the model's embedding layer
    embeddings = model.model.embed_tokens(input_ids)  # Shape: [batch_size, seq_len, hidden_size]

    # Now we can pass these embeddings to the forward method
    # This will skip the embedding layer and start at layer 0
    output = model.forward(inputs_embeds=embeddings)

    # You can compare this with regular forward pass
    regular_output = model.forward(input_ids=input_ids)

    # The outputs should be similar (though not exactly the same due to different code paths)
    print("Embeddings shape:", embeddings.shape)
    print("Output from inputs_embeds shape:", output.logits.shape)
    print("Output from input_ids shape:", regular_output.logits.shape)
    # Verify outputs are numerically equivalent
    assert torch.allclose(output.logits, regular_output.logits, rtol=1e-4, atol=1e-4), "Outputs from inputs_embeds and input_ids should be equivalent"
    