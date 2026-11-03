# Project author: Nairouz Mrabah
# License: MIT for original project contributions (see LICENSE at the repository root).
# Upstream sources and their original credits are preserved in this module.
# Microsoft LoRA terms: licenses/LoRA-MIT.txt; other upstream terms still apply.
# Third-party code retains its original authorship and license.
# See THIRD_PARTY_NOTICES.md at the repository root.

import os

import torch
import torch.nn as nn
from typing import Dict
from .layers import LoRALayer, PlainMultiheadAttentionLoRA, LinearLoRA


INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None: 
    """
    Freezes all parameters except for LoRA parameters and optionally certain biases.
    
    Args:
    - model (nn.Module): The PyTorch model containing LoRA layers.
    - bias (str): Specifies which biases should be trainable:
        - 'none': No biases are trainable (default).
        - 'all': All biases in the model are trainable.
        - 'lora_only': Only biases inside LoRA layers are trainable.

    Returns:
    - None: The function modifies the model in-place.
    """
    
    trainable_params = []  # Store names and shapes of trainable parameters

    # Step 1: Freeze all parameters except LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_' not in name:  
            param.requires_grad = False  # Freeze all non-LoRA parameters
        else:
            param.requires_grad = True   # Keep LoRA parameters trainable
            trainable_params.append((name, tuple(param.shape)))  # Store trainable parameter name and shape

    # Step 2: Handle biases if necessary
    if bias == 'all':
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True  
                trainable_params.append((name, tuple(param.shape)))

    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and hasattr(m, 'bias') and m.bias is not None:
                m.bias.requires_grad = True  
                trainable_params.append((f"{m}.bias", tuple(m.bias.shape)))

    # ✅ Print trainable parameters
    print("\n✅ Trainable Parameters in LoRA CLIP:")
    for param_name, shape in trainable_params:
        print(f"{param_name}: {shape}")
        
'''
def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """
    Freezes all parameters in the model except for LoRA parameters and optionally certain biases.

    Args:
    - model (nn.Module): The PyTorch model containing LoRA layers and other parameters.
    - bias (str): Specifies which biases should be trainable:
        - 'none': No biases are trainable (default).
        - 'all': All biases in the model are trainable.
        - 'lora_only': Only biases inside LoRA layers are trainable.
    
    Returns:
    - None: The function modifies the model in-place by setting the `requires_grad` attribute.
    """
    
    # Step 1: Freeze all parameters that are not related to LoRA
    for n, p in model.named_parameters():  # Iterate over all named parameters in the model
        if 'lora_' not in n:  # If the parameter name does not include 'lora_', freeze it
            p.requires_grad = False  # Prevent gradients from being computed for this parameter

    # Step 2: Handle the 'bias' argument
    if bias == 'none':  
        # If bias handling is set to 'none', return immediately without making biases trainable
        return

    elif bias == 'all':  
        # If all biases in the model should be trainable
        for n, p in model.named_parameters():  # Iterate over all named parameters
            if 'bias' in n:  # Check if the parameter name contains 'bias'
                p.requires_grad = True  # Make biases trainable

    elif bias == 'lora_only':  
        # If only biases within LoRA layers should be trainable
        for m in model.modules():  # Iterate over all modules (layers) in the model
            if isinstance(m, LoRALayer) and hasattr(m, 'bias') and m.bias is not None: # Ensure the bias is not None
                m.bias.requires_grad = True  # Make the bias within this LoRA layer trainable
    else:
        # Raise an error if an unsupported value for the 'bias' argument is provided
        raise NotImplementedError(f"The bias option '{bias}' is not implemented. Use 'none', 'all', or 'lora_only'.")
'''
        
def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    """
    Extracts the state dictionary of a PyTorch model, focusing on LoRA parameters
    and optionally including biases based on the specified option.

    Args:
    - model (nn.Module): The PyTorch model containing LoRA layers and other parameters.
    - bias (str): Specifies which biases should be included:
        - 'none': Includes only LoRA parameters (default).
        - 'all': Includes both LoRA parameters and all biases in the model.
        - 'lora_only': Includes LoRA parameters and only biases associated with LoRA layers.

    Returns:
    - Dict[str, torch.Tensor]: A dictionary containing the selected parameters.
    """
    # Get the full state dictionary of the model, containing all parameters.
    my_state_dict = model.state_dict()

    # Case 1: If no biases should be included, return only the LoRA parameters.
    if bias == 'none':
        # Filter the state dictionary to include only keys containing 'lora_'.
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}

    # Case 2: If all biases in the model should be included.
    elif bias == 'all':
        # Filter the state dictionary to include keys containing 'lora_' or 'bias'.
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}

    # Case 3: If only biases associated with LoRA layers should be included.
    elif bias == 'lora_only':
        # Initialize an empty dictionary to store the relevant parameters.
        to_return = {}

        # Iterate through all keys in the state dictionary.
        for k in my_state_dict:
            if 'lora_' in k:  # Check if the key corresponds to a LoRA parameter.
                # Add the LoRA parameter to the result dictionary.
                to_return[k] = my_state_dict[k]

                # Construct the associated bias key by removing 'lora_' and appending 'bias'.
                bias_name = k.split('lora_')[0] + 'bias'

                # If the associated bias exists in the state dictionary, add it.
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]

        # Return the dictionary containing LoRA parameters and their associated biases.
        return to_return

    # Raise an error if the bias option is not recognized.
    else:
        raise NotImplementedError(f"The bias option '{bias}' is not implemented. Use 'none', 'all', or 'lora_only'.")

def get_lora_parameters(model, bias='none'):
    """
    Extracts trainable parameters from a PyTorch model, focusing on LoRA parameters
    and optionally including biases based on the specified option.

    Args:
    - model (nn.Module): The PyTorch model containing parameters and LoRA layers.
    - bias (str): Specifies which biases to include:
        - 'none': Only LoRA parameters are included (default).
        - 'all': LoRA parameters and all biases are included.
        - 'lora_only': LoRA parameters and only biases directly associated with LoRA layers are included.

    Returns:
    - List[torch.Tensor]: A list of trainable parameter tensors based on the specified criteria.
    """
    # Initialize an empty list to store the selected parameters.
    params = []

    # Iterate over all named parameters in the model.
    for name, param in model.named_parameters():

        # Case 1: 'none' - Include only LoRA parameters.
        if bias == 'none':
            if 'lora_' in name:  # Check if the parameter name contains 'lora_'.
                params.append(param)  # Add the parameter to the list.

        # Case 2: 'all' - Include both LoRA parameters and all biases.
        elif bias == 'all':
            if 'lora_' in name or 'bias' in name:  # Check if the name contains 'lora_' or 'bias'.
                params.append(param)  # Add the parameter to the list.

        # Case 3: 'lora_only' - Include LoRA parameters and their associated biases.
        elif bias == 'lora_only':
            if 'lora_' in name:  # Check if the parameter name contains 'lora_'.
                params.append(param)  # Add the LoRA parameter to the list.

                # Construct the name of the associated bias parameter.
                bias_name = name.split('lora_')[0] + 'bias'

                # Check if the associated bias exists in the model's state dictionary.
                if bias_name in model.state_dict():
                    # Retrieve the associated bias parameter using the bias name.
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)  # Add the bias parameter to the list.

        # Handle invalid bias values by raising an error.
        else:
            raise NotImplementedError(
                f"The bias option '{bias}' is not implemented. Use 'none', 'all', or 'lora_only'."
            )

    # Return the list of selected parameters.
    return params

'''
def apply_lora(args, clip_model):
    """
    Replaces specific multi-head attention layers in the CLIP model's text and/or vision encoders
    with LoRA-enhanced multi-head attention layers.

    Args:
    - args: An object containing configuration options, such as:
        - args.encoder: Specifies which encoder to modify ('text', 'vision', or 'both').
        - args.position: Specifies which attention layers to replace ('all', 'top3', etc.).
        - args.params: Attention matrices to apply LoRA ('q', 'k', 'v').
        - args.rank: Low-rank dimension for LoRA.
        - args.alpha: Scaling factor for LoRA.
        - args.dropout_rate: Dropout rate before the LoRA module.
        - args.backbone: Backbone architecture for the vision model (e.g., 'ViT-B/16').
    - clip_model: The CLIP model (PyTorch nn.Module) to be modified.

    Returns:
    - list_lora_layers: A list of all LoRA layers added to the model for easy tracking.
    """
    # Initialize an empty list to keep track of the LoRA layers added to the model.
    list_lora_layers = []

    # Apply LoRA to the text encoder if 'text' or 'both' is specified in args.encoder.
    if args.encoder == 'text' or args.encoder == 'both':
        # Get the indices of the residual blocks where LoRA will be applied for the text encoder.
        indices = INDEX_POSITIONS_TEXT[args.position]

        # Access the text encoder of the CLIP model.
        text_encoder = clip_model.transformer

        # Iterate through the residual blocks of the text encoder.
        for i, block in enumerate(text_encoder.resblocks):
            print(f"Residual Attention Block {i}: {block}")  # Print the current block being processed.

            # Check if the current block index is in the specified indices for LoRA application.
            if i in indices:
                # Iterate through all named submodules in the current residual block.
                for name, submodule in block.named_children():
                    # Check if the submodule is a multi-head attention layer.
                    if isinstance(submodule, nn.MultiheadAttention):
                        # Create a new LoRA-enabled multi-head attention layer.
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule,  # The original multi-head attention layer.
                            enable_lora=args.params,  # Specifies which matrices ('q', 'k', 'v') get LoRA.
                            r=args.rank,  # Low-rank dimension for LoRA.
                            lora_alpha=args.alpha,  # Scaling factor for LoRA.
                            dropout_rate=args.dropout_rate  # Dropout rate applied before the LoRA module.
                        )

                        # Replace the original multi-head attention layer with the new LoRA-enabled layer.
                        setattr(block, name, new_multi_head_lora)

                        # Add the new LoRA layer to the list for tracking.
                        list_lora_layers.append(new_multi_head_lora)

    # Apply LoRA to the vision encoder if 'vision' or 'both' is specified in args.encoder.
    if args.encoder == 'vision' or args.encoder == 'both':
        # Get the indices of the residual blocks where LoRA will be applied for the vision encoder.
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]

        # Access the vision encoder of the CLIP model.
        vision_encoder = clip_model.visual.transformer

        # Iterate through the residual blocks of the vision encoder.
        for i, block in enumerate(vision_encoder.resblocks):
            print(f"Residual Attention Block {i}: {block}")  # Print the current block being processed.

            # Check if the current block index is in the specified indices for LoRA application.
            if i in indices:
                # Iterate through all named submodules in the current residual block.
                for name, submodule in block.named_children():
                    # Check if the submodule is a multi-head attention layer.
                    if isinstance(submodule, nn.MultiheadAttention):
                        # Create a new LoRA-enabled multi-head attention layer.
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule,  # The original multi-head attention layer.
                            enable_lora=args.params,  # Specifies which matrices ('q', 'k', 'v') get LoRA.
                            r=args.rank,  # Low-rank dimension for LoRA.
                            lora_alpha=args.alpha,  # Scaling factor for LoRA.
                            dropout_rate=args.dropout_rate  # Dropout rate applied before the LoRA module.
                        )

                        # Replace the original multi-head attention layer with the new LoRA-enabled layer.
                        setattr(block, name, new_multi_head_lora)

                        # Add the new LoRA layer to the list for tracking.
                        list_lora_layers.append(new_multi_head_lora)

    # Return the list of all added LoRA layers for tracking and further operations.
    return list_lora_layers
'''

def apply_lora(args, clip_model): 
    """
    Extends LoRA integration to both attention and MLP layers in the CLIP model.

    Applies LoRA to:
    - Multi-head attention (`attn.in_proj_weight`, `attn.out_proj.weight`)
    - MLP layers (`mlp.c_fc.weight`, `mlp.c_proj.weight`)

    Excludes:
    - Bias terms (`attn.in_proj_bias`, `attn.out_proj.bias`, `mlp.c_fc.bias`, `mlp.c_proj.bias`)
    - LayerNorm parameters (`ln_1.weight`, `ln_1.bias`, `ln_2.weight`, `ln_2.bias`)

    Args:
        args: Configuration specifying:
            - `encoder`: 'text', 'vision', or 'both'
            - `position`: Specifies which layers to modify ('all', 'top3', etc.)
            - `params`: Specifies which attention matrices to apply LoRA ('q', 'k', 'v', 'o')
            - `rank`: Low-rank dimension for LoRA
            - `alpha`: Scaling factor for LoRA
            - `dropout_rate`: Dropout rate before LoRA
            - `backbone`: Backbone architecture for vision model (e.g., 'ViT-B/16')
        clip_model: The CLIP model to be modified.

    Returns:
        list_lora_layers: List of LoRA layers added to the model.
    """

    list_lora_layers = []  # Track all added LoRA layers

    # Apply LoRA to the text encoder if specified
    if args.encoder in ['text', 'both']:
        indices = INDEX_POSITIONS_TEXT[args.position]
        text_encoder = clip_model.transformer

        for i, block in enumerate(text_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    # Apply LoRA to multi-head attention
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule,
                            enable_lora=args.params + ['o'],  # Ensure 'o' is included for attn.out_proj
                            r=args.rank,
                            lora_alpha=args.alpha,
                            dropout_rate=args.dropout_rate
                        )
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

                    # Apply LoRA to MLP layers (only weight matrices)
                    elif name == "mlp":
                        for mlp_name, mlp_layer in submodule.named_children():
                            if isinstance(mlp_layer, nn.Linear) and "bias" not in mlp_name:
                                new_mlp_lora = LinearLoRA(
                                    mlp_layer,
                                    r=args.rank,
                                    lora_alpha=args.alpha,
                                    dropout_rate=args.dropout_rate
                                )
                                setattr(submodule, mlp_name, new_mlp_lora)
                                list_lora_layers.append(new_mlp_lora)

                    # Skip LayerNorm layers (ln_1, ln_2)
                    elif name.startswith("ln__"):
                        continue

    # Apply LoRA to the vision encoder if specified
    if args.encoder in ['vision', 'both']:
        indices = INDEX_POSITIONS_VISION[args.backbone][args.position]
        vision_encoder = clip_model.visual.transformer

        for i, block in enumerate(vision_encoder.resblocks):
            if i in indices:
                for name, submodule in block.named_children():
                    # Apply LoRA to multi-head attention
                    if isinstance(submodule, nn.MultiheadAttention):
                        new_multi_head_lora = PlainMultiheadAttentionLoRA(
                            submodule,
                            enable_lora=args.params + ['o'],  # Ensure 'o' is included for attn.out_proj
                            r=args.rank,
                            lora_alpha=args.alpha,
                            dropout_rate=args.dropout_rate
                        )
                        setattr(block, name, new_multi_head_lora)
                        list_lora_layers.append(new_multi_head_lora)

                    # Apply LoRA to MLP layers (only weight matrices)
                    elif name == "mlp":
                        for mlp_name, mlp_layer in submodule.named_children():
                            if isinstance(mlp_layer, nn.Linear) and "bias" not in mlp_name:
                                new_mlp_lora = LinearLoRA(
                                    mlp_layer,
                                    r=args.rank,
                                    lora_alpha=args.alpha,
                                    dropout_rate=args.dropout_rate
                                )
                                setattr(submodule, mlp_name, new_mlp_lora)
                                list_lora_layers.append(new_mlp_lora)

                    # Skip LayerNorm layers (ln_1, ln_2)
                    elif name.startswith("ln_"):
                        continue

    return list_lora_layers


def save_lora(args, list_lora_layers):
    """
    Saves LoRA weights for both attention and MLP layers.

    Args:
        args: Configuration containing model details.
        list_lora_layers: List of LoRA layers applied to the model.

    Saves:
        - LoRA weights for attention layers (`q_proj`, `k_proj`, `v_proj`, `proj`).
        - LoRA weights for MLP layers (`mlp.c_fc`, `mlp.c_proj`).
        - Metadata to ensure compatibility during loading.
    """
    weights = {}

    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}

        # Attention weights
        if hasattr(layer, "q_proj") and hasattr(layer.q_proj, "w_lora_A"):
            layer_weights["q_proj"] = {
                "w_lora_A": layer.q_proj.w_lora_A.data,
                "w_lora_B": layer.q_proj.w_lora_B.data
            }
        if hasattr(layer, "k_proj") and hasattr(layer.k_proj, "w_lora_A"):
            layer_weights["k_proj"] = {
                "w_lora_A": layer.k_proj.w_lora_A.data,
                "w_lora_B": layer.k_proj.w_lora_B.data
            }
        if hasattr(layer, "v_proj") and hasattr(layer.v_proj, "w_lora_A"):
            layer_weights["v_proj"] = {
                "w_lora_A": layer.v_proj.w_lora_A.data,
                "w_lora_B": layer.v_proj.w_lora_B.data
            }
        if hasattr(layer, "proj") and hasattr(layer.proj, "w_lora_A"):
            layer_weights["proj"] = {
                "w_lora_A": layer.proj.w_lora_A.data,
                "w_lora_B": layer.proj.w_lora_B.data
            }

        # MLP weights (excluding bias)
        if hasattr(layer, "c_fc") and hasattr(layer.c_fc, "w_lora_A"):
            layer_weights["c_fc"] = {
                "w_lora_A": layer.c_fc.w_lora_A.data,
                "w_lora_B": layer.c_fc.w_lora_B.data
            }
        if hasattr(layer, "c_proj") and hasattr(layer.c_proj, "w_lora_A"):
            layer_weights["c_proj"] = {
                "w_lora_A": layer.c_proj.w_lora_A.data,
                "w_lora_B": layer.c_proj.w_lora_B.data
            }

        weights[f"layer_{i}"] = layer_weights

    # Metadata for compatibility checks
    metadata = {
        "r": args.rank,
        "alpha": args.alpha,
        "encoder": args.encoder,
        "params": args.params,
        "position": args.position
    }

    save_data = {
        "weights": weights,
        "metadata": metadata
    }

    backbone = args.backbone.replace("/", "-")
    opt = args.optimizer.lower()  
    fname = f"dataset_{args.dataset}_shots_{args.shots}_backbone_{backbone}_opt_{opt}_seed_{args.seed}.pt"
    save_path = os.path.join(args.save_path, fname)
    torch.save(save_data, save_path)
    print(f"✅ LoRA weights saved to {save_path}")

def load_lora(args, list_lora_layers):
    """
    Loads LoRA weights into both attention and MLP layers.

    Args:
        args: Configuration specifying the model and LoRA details.
        list_lora_layers: List of LoRA layers where weights will be loaded.

    Validates:
        - Rank (`r`), Alpha (`alpha`), Encoder type, Params, Position.
    """    
    backbone = args.backbone.replace("/", "-")
    opt = args.optimizer.lower()  
    fname = f"dataset_{args.dataset}_shots_{args.shots}_backbone_{backbone}_opt_{opt}_seed_{args.seed}.pt"    
    load_path = os.path.join(args.save_path, fname) 

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"❌ File {load_path} does not exist.")

    try:
        loaded_data = torch.load(load_path, weights_only=True, map_location="cpu")
    except TypeError:
        loaded_data = torch.load(load_path)
    metadata = loaded_data["metadata"]
    
    # Validate metadata before loading weights
    _key2arg = {"r": "rank", "alpha": "alpha", "encoder": "encoder", "params": "params", "position": "position"}

    for mkey, arg_name in _key2arg.items():
        if metadata[mkey] != getattr(args, arg_name):
            raise ValueError(f"❌ {mkey} mismatch: expected {getattr(args, arg_name)}, found {metadata[mkey]}")
        
    weights = loaded_data["weights"]
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f"layer_{i}"]

        # Load attention weights
        if "q_proj" in layer_weights:
            layer.q_proj.w_lora_A.data.copy_(layer_weights["q_proj"]["w_lora_A"])
            layer.q_proj.w_lora_B.data.copy_(layer_weights["q_proj"]["w_lora_B"])
        if "k_proj" in layer_weights:
            layer.k_proj.w_lora_A.data.copy_(layer_weights["k_proj"]["w_lora_A"])
            layer.k_proj.w_lora_B.data.copy_(layer_weights["k_proj"]["w_lora_B"])
        if "v_proj" in layer_weights:
            layer.v_proj.w_lora_A.data.copy_(layer_weights["v_proj"]["w_lora_A"])
            layer.v_proj.w_lora_B.data.copy_(layer_weights["v_proj"]["w_lora_B"])
        if "proj" in layer_weights:
            layer.proj.w_lora_A.data.copy_(layer_weights["proj"]["w_lora_A"])
            layer.proj.w_lora_B.data.copy_(layer_weights["proj"]["w_lora_B"])

        # Load MLP weights (excluding bias)
        if "c_fc" in layer_weights:
            layer.c_fc.w_lora_A.data.copy_(layer_weights["c_fc"]["w_lora_A"])
            layer.c_fc.w_lora_B.data.copy_(layer_weights["c_fc"]["w_lora_B"])
        if "c_proj" in layer_weights:
            layer.c_proj.w_lora_A.data.copy_(layer_weights["c_proj"]["w_lora_A"])
            layer.c_proj.w_lora_B.data.copy_(layer_weights["c_proj"]["w_lora_B"])

    print(f"✅ LoRA weights loaded from {load_path}")


'''
def save_lora(args, list_lora_layers):
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        if 'q' in args.params:
            layer_weights['q_proj'] = {
                'w_lora_A': layer.q_proj.w_lora_A.data,
                'w_lora_B': layer.q_proj.w_lora_B.data
            }
        if 'k' in args.params:
            layer_weights['k_proj'] = {
                'w_lora_A': layer.k_proj.w_lora_A.data,
                'w_lora_B': layer.k_proj.w_lora_B.data
            }
        if 'v' in args.params:
            layer_weights['v_proj'] = {
                'w_lora_A': layer.v_proj.w_lora_A.data,
                'w_lora_B': layer.v_proj.w_lora_B.data
            }
        if 'o' in args.params:
            layer_weights['proj'] = {
                'w_lora_A': layer.proj.w_lora_A.data,
                'w_lora_B': layer.proj.w_lora_B.data
            }

        weights[f'layer_{i}'] = layer_weights

    metadata = {
        'r': args.rank,
        'alpha': args.alpha,
        'encoder': args.encoder,
        'params': args.params,
        'position': args.position
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }

    # to manage names like ViT-B/16
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    save_dir = f'{args.save_path}/{backbone}/{args.dataset}/{args.shots}shots/seed{args.seed}'
    os.makedirs(save_dir, exist_ok=True)

    save_path = f'{save_dir}/{args.filename}.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')

def load_lora(args, list_lora_layers):
    # to manage names like ViT-B/16
    backbone = args.backbone.replace('/', '').replace('-', '').lower()
    load_path = f'{args.save_path}/{backbone}/{args.dataset}/{args.shots}shots/seed{args.seed}/{args.filename}.pt'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    if metadata['r'] != args.rank:
        raise ValueError(
            f"r mismatch: expected {args.rank}, found {metadata['r']}")
    if metadata['alpha'] != args.alpha:
        raise ValueError(
            f"alpha mismatch: expected {args.alpha}, found {metadata['alpha']}")
    if metadata['encoder'] != args.encoder:
        raise ValueError(
            f"Encoder mismatch: expected {args.encoder}, found {metadata['encoder']}")
    if metadata['params'] != args.params:
        raise ValueError(
            f"Params mismatch: expected {args.params}, found {metadata['params']}")
    if metadata['position'] != args.position:
        raise ValueError(
            f"Position mismatch: expected {args.position}, found {metadata['position']}")

    weights = loaded_data['weights']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights[f'layer_{i}']
        if 'q' in args.params and 'q_proj' in layer_weights:
            layer.q_proj.w_lora_A.data.copy_(
                layer_weights['q_proj']['w_lora_A'])
            layer.q_proj.w_lora_B.data.copy_(
                layer_weights['q_proj']['w_lora_B'])
        if 'k' in args.params and 'k_proj' in layer_weights:
            layer.k_proj.w_lora_A.data.copy_(
                layer_weights['k_proj']['w_lora_A'])
            layer.k_proj.w_lora_B.data.copy_(
                layer_weights['k_proj']['w_lora_B'])
        if 'v' in args.params and 'v_proj' in layer_weights:
            layer.v_proj.w_lora_A.data.copy_(
                layer_weights['v_proj']['w_lora_A'])
            layer.v_proj.w_lora_B.data.copy_(
                layer_weights['v_proj']['w_lora_B'])
        if 'o' in args.params and 'proj' in layer_weights:
            layer.proj.w_lora_A.data.copy_(layer_weights['proj']['w_lora_A'])
            layer.proj.w_lora_B.data.copy_(layer_weights['proj']['w_lora_B'])

    print(f'LoRA weights loaded from {load_path}')
'''