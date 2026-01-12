import importlib

__attributes = {
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    'SLatEncoder': 'structured_latent_vae',
    'SLatGaussianDecoder': 'structured_latent_vae',
    'SLatFlowModel': 'structured_latent_flow',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name not in ["SLatMeshDecoder", "SLatRadianceFieldDecoder"]:
            if name in __attributes:
                module_name = __attributes[name]
                module = importlib.import_module(f".{module_name}", __name__)
                globals()[name] = getattr(module, name)
            elif name in __submodules:
                module = importlib.import_module(f".{name}", __name__)
                globals()[name] = module
            else:
                raise AttributeError(f"module {__name__} has no attribute {name}")
            return globals()[name]


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    import sys
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
        print(f"[Trellis] Loading model from local path: {path}", file=sys.stderr, flush=True)
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        print(f"[Trellis] Downloading model config: {model_name}.json from {repo_id}...", file=sys.stderr, flush=True)
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        print(f"[Trellis] Downloading model weights: {model_name}.safetensors from {repo_id} (this may take several minutes)...", file=sys.stderr, flush=True)
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")
        print(f"[Trellis] Model download complete: {model_name}", file=sys.stderr, flush=True)

    with open(config_file, 'r') as f:
        config = json.load(f)

    if config['name'] not in ["SLatMeshDecoder", "SLatRadianceFieldDecoder"]:
        model = __getattr__(config['name'])(**config['args'], **kwargs)
        model.load_state_dict(load_file(model_file))
    else:
        model = None

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_vae import SLatEncoder, SLatGaussianDecoder
    from .structured_latent_flow import SLatFlowModel
