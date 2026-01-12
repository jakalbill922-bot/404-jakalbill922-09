from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, TypeAlias, Union
from PIL import Image

from schemas.overridable import OverridableModel

class TrellisMode(str, Enum):
    STOCHASTIC: str = 'stochastic'
    MULTIDIFFUSION: str = 'multidiffusion'

class TrellisParams(OverridableModel):
    """Trellis parameters with automatic fallback to settings."""
    sparse_structure_steps: int
    sparse_structure_cfg_strength: float
    slat_steps: int
    slat_cfg_strength: float
    mode: TrellisMode = TrellisMode.STOCHASTIC
    num_oversamples: int = 1
    dynamic_steps: bool
    
    @classmethod
    def from_settings(cls, settings) -> "TrellisParams":
        return cls(
            sparse_structure_steps = settings.sparse_structure_steps,
            sparse_structure_cfg_strength = settings.sparse_structure_cfg_strength,
            slat_steps = settings.slat_steps,
            slat_cfg_strength = settings.slat_cfg_strength,
            num_oversamples = settings.num_oversamples,
            mode = settings.mode,
            dynamic_steps = settings.dynamic_steps,
        )

TrellisParamsOverrides: TypeAlias = TrellisParams.Overrides

@dataclass
class TrellisRequest:
    """Request for Trellis 3D generation (internal use only)."""
    image: Union[Image.Image, Iterable[Image.Image]]
    seed: int
    params: Optional[TrellisParamsOverrides] = None


@dataclass(slots=True)
class TrellisResult:
    """Result from Trellis 3D generation."""
    ply_file: bytes | None = None


