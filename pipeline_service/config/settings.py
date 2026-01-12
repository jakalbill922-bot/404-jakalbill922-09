from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings

config_dir = Path(__file__).parent
config_file_dir = Path(__file__).parent.parent / "configuration.yaml"


class APIConfig(BaseModel):
    """API configuration"""
    api_title: str = "3D Generation pipeline Service"
    host: str = "0.0.0.0"
    port: int = 10006

class OutputConfig(BaseModel):
    """Output configuration"""
    save_generated_files: bool = False
    send_generated_files: bool = False
    compression: bool = False
    output_dir: Path = Path("generated_outputs")
    
class TrellisConfig(BaseModel):
    """Trellis model configuration"""
    model_id: str = "jetx/trellis-image-large"
    sparse_structure_steps: int = 8
    sparse_structure_cfg_strength: float = 5.75
    slat_steps: int = 20
    slat_cfg_strength: float = 2.4
    num_oversamples: int = 3
    mode: str = "stochastic"
    multiview: bool = True  # If True, generate multiple views; if False, only single view
    gpu: int = 0
    dynamic_steps: bool = True

class QwenConfig(BaseModel):
    """Qwen model configuration"""
    base_model_path: str = "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"
    model_path: str = "Qwen/Qwen-Image-Edit-2509"
    lora_path: str = "lightx2v/Qwen-Image-Lightning"
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 8
    true_cfg_scale: float = 1.0
    prompt_path_base: Path = Path("prompts") / "qwen_edit_prompt_base.json"
    prompt_path_multistage: Path = Path("prompts") / "qwen_edit_prompt_multistage_v5.json"
    gpu: int = 0
    dtype: str = "bf16"

class BackgroundRemovalConfig(BaseModel):
    """Background removal configuration"""
    model_id: str = "PramaLLC/BEN2"
    input_image_size: tuple[int, int] = (1024, 1024)
    output_image_size: tuple[int, int] = (518, 518)
    padding_percentage: float = 0.2
    limit_padding: bool = True
    gpu: int = 0

class SettingsConf(BaseSettings):
    """Main settings class"""
    api: APIConfig = APIConfig()
    output: OutputConfig
    trellis: TrellisConfig
    qwen: QwenConfig
    background_removal: BackgroundRemovalConfig
        

def _load_yml_config(path: Path):
    """Classmethod returns YAML config"""
    try:
        return yaml.safe_load(path.read_text())

    except FileNotFoundError as error:
        message = "Error: yml config file not found."
        raise FileNotFoundError(error, message) from error

data_yaml = _load_yml_config(config_file_dir)
settings = SettingsConf.model_validate(data_yaml)
