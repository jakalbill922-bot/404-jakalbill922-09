
import json
import yaml
from safetensors import safe_open
from pydantic import RootModel
from os import PathLike
from pathlib import Path
from typing import Dict, Optional
from config.settings import QwenConfig
from modules.image_edit.prompting import EmbeddedPrompting, Prompting, TextPrompting


class PromptingLibrary(RootModel):
    root: Dict[str, TextPrompting]

    @property
    def promptings(self):
        return self.root

    @classmethod
    def from_file(cls, path: PathLike):
        # Load the promptings from the specified path. This method should handle both .json and .yaml files.
        with open(path, "r") as f:
            data = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
        return cls.model_validate(data)