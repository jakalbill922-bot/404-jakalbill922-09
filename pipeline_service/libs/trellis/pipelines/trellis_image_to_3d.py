from PIL import Image
from typing import *
from contextlib import contextmanager


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp


class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(TrellisImageTo3DPipeline, TrellisImageTo3DPipeline).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']
        new_pipeline.slat_normalization = args['slat_normalization']
        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load('facebookresearch/dinov2', name, pretrained=True)
        dinov2_model.eval()
        self.models['image_cond_model'] = dinov2_model

        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """

        output_np = np.array(input)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = input.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(self, image: Union[torch.Tensor, list[Image.Image]]) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.Resampling.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        
        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models['image_cond_model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])

        return patchtokens
        
    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            'cond': cond,
            'neg_cond': neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples
        
        # Decode occupancy latent
        decoder = self.models['sparse_structure_decoder']
        coords = torch.argwhere(decoder(z_s)>0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'gaussian' in formats:
            ret['gaussian'] = self.models['slat_decoder_gs'](slat)

        return ret
    
    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        flow_model = self.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True
        ).samples

        std = torch.tensor(self.slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        # seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian'],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])

        # torch.manual_seed(seed)

        coords = self.sample_sparse_structure(cond, num_samples, sparse_structure_sampler_params)
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @torch.no_grad()
    def run_multi_image(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["gaussian"],
        preprocess_image: bool = True,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
        num_oversamples: int = 1,
    ) -> dict:
        """
        Run the pipeline with multiple images as condition

        Args:
            images (List[Image.Image]): The multi-view images of the assets
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond["neg_cond"] = cond["neg_cond"][:1]
        torch.manual_seed(seed)
        num_oversamples = max(num_samples, num_oversamples)
        ss_steps = {
            **self.sparse_structure_sampler_params,
            **sparse_structure_sampler_params,
        }.get("steps")
        with self.inject_sampler_multi_image(
            "sparse_structure_sampler", len(images), ss_steps, mode=mode
        ):
            coords = self.sample_sparse_structure(
                cond, num_oversamples, sparse_structure_sampler_params
            )
            coords = (
                coords
                if num_oversamples <= num_samples
                else self.select_coords(coords, num_samples)
            )
        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get("steps")
        with self.inject_sampler_multi_image(
            "slat_sampler", len(images), slat_steps, mode=mode
        ):
            slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    
    def select_coords(self, coords, num_samples):
        """
        Select n smallest sparse structures in terms of number of voxels
        """
        counts = coords[:, 0].unique(return_counts=True)[-1]
        selected_coords = sorted(
            coords[:, 1:].split(tuple(counts.tolist())), key=lambda x: len(x)
        )[:num_samples]
        sizes = torch.tensor(tuple(len(coo) for coo in selected_coords))
        selected_coords = torch.cat(selected_coords, dim=0)
        indices = (
            torch.arange(num_samples)
            .repeat_interleave(sizes)
            .unsqueeze(-1)
            .to(selected_coords.device, selected_coords.dtype)
        )
        selected_coords = torch.cat((indices, selected_coords), dim=1)
        return selected_coords

    @contextmanager
    def inject_sampler_multi_image(
        self,
        sampler_name: str,
        num_images: int,
        num_steps: int,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
    ):
        """
        Inject a sampler with multiple images as condition.

        Args:
            sampler_name (str): The name of the sampler to inject.
            num_images (int): The number of images to condition on.
            num_steps (int): The number of steps to run the sampler for.
        """
        sampler = getattr(self, sampler_name)
        setattr(sampler, f"_old_inference_model", sampler._inference_model)

        if mode == "stochastic":
            if num_images > num_steps:
                print(
                    f"\033[93mWarning: number of conditioning images is greater than number of steps for {sampler_name}. "
                    "This may lead to performance degradation.\033[0m"
                )

            cond_indices = (np.arange(num_steps) % num_images).tolist()

            def _new_inference_model(self, model, x_t, t, cond, **kwargs):
                cond_idx = cond_indices.pop(0)
                cond_i = cond[cond_idx : cond_idx + 1]
                # Expand conditioning to match x_t batch size for cross-attention
                batch_size = x_t.shape[0]
                if cond_i.shape[0] != batch_size:
                    cond_i = cond_i.expand(batch_size, -1, -1)
                # Also expand neg_cond if present in kwargs
                if 'neg_cond' in kwargs and kwargs['neg_cond'] is not None:
                    neg_cond = kwargs['neg_cond']
                    if neg_cond.shape[0] != batch_size:
                        kwargs['neg_cond'] = neg_cond.expand(batch_size, -1, -1)
                return self._old_inference_model(model, x_t, t, cond=cond_i, **kwargs)

        elif mode == "multidiffusion":
            from .samplers import FlowEulerSampler

            def _new_inference_model(
                self,
                model,
                x_t,
                t,
                cond,
                neg_cond,
                cfg_strength,
                cfg_interval,
                **kwargs,
            ):
                batch_size = x_t.shape[0]
                if cfg_interval[0] <= t <= cfg_interval[1]:
                    preds = []
                    for i in range(len(cond)):
                        cond_i = cond[i : i + 1]
                        if cond_i.shape[0] != batch_size:
                            cond_i = cond_i.expand(batch_size, -1, -1)
                        preds.append(
                            FlowEulerSampler._inference_model(
                                self, model, x_t, t, cond_i, **kwargs
                            )
                        )
                    pred = sum(preds) / len(preds)
                    neg_cond_exp = neg_cond if neg_cond.shape[0] == batch_size else neg_cond.expand(batch_size, -1, -1)
                    neg_pred = FlowEulerSampler._inference_model(
                        self, model, x_t, t, neg_cond_exp, **kwargs
                    )
                    return (1 + cfg_strength) * pred - cfg_strength * neg_pred
                else:
                    preds = []
                    for i in range(len(cond)):
                        cond_i = cond[i : i + 1]
                        if cond_i.shape[0] != batch_size:
                            cond_i = cond_i.expand(batch_size, -1, -1)
                        preds.append(
                            FlowEulerSampler._inference_model(
                                self, model, x_t, t, cond_i, **kwargs
                            )
                        )
                    pred = sum(preds) / len(preds)
                    return pred

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        sampler._inference_model = _new_inference_model.__get__(sampler, type(sampler))

        yield

        sampler._inference_model = sampler._old_inference_model
        delattr(sampler, f"_old_inference_model")

    def _interpolate_value(
        self, num_voxels: int, ranges: List[tuple[int, float]]
    ) -> float:
        """
        Calculate interpolated value based on voxel count with linear interpolation between ranges.
        
        Args:
            num_voxels (int): Number of occupied voxels.
            ranges (List[tuple[int, float]]): List of (voxel_threshold, value) tuples.
            
        Returns:
            float: The interpolated value.
        """
        # Sort ranges by voxel count
        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        
        # Below the lowest threshold - use the lowest value
        if num_voxels <= sorted_ranges[0][0]:
            return sorted_ranges[0][1]
        
        # Above the highest threshold - use the highest value
        if num_voxels >= sorted_ranges[-1][0]:
            return sorted_ranges[-1][1]
        
        # Linear interpolation between ranges
        for i in range(len(sorted_ranges) - 1):
            low_voxel, low_val = sorted_ranges[i]
            high_voxel, high_val = sorted_ranges[i + 1]
            
            if low_voxel <= num_voxels < high_voxel:
                t = (num_voxels - low_voxel) / (high_voxel - low_voxel)
                return low_val + t * (high_val - low_val)
        
        return 1.0  # fallback

    @torch.no_grad()
    def run_multi_image_with_voxel_count(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["gaussian"],
        preprocess_image: bool = True,
        mode: Literal["stochastic", "multidiffusion"] = "stochastic",
        num_oversamples: int = 1,
        voxel_ranges: List[tuple[int, float]] = None,
        cfg_ranges: List[tuple[int, float]] = None,
        dynamic_steps: bool = True,
    ) -> tuple[dict, int, int, float]:
        """
        Run the pipeline with multiple images as condition and adaptive texture steps based on voxel count.
        
        Uses a multi-threshold system with linear interpolation for smoother transitions
        between quality levels based on object complexity (voxel count).

        Args:
            images (List[Image.Image]): The multi-view images of the assets.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
            voxel_ranges (List[tuple[int, float]]): List of (voxel_threshold, step_multiplier) tuples.
                Sorted ascending by voxel count. Linear interpolation is applied between thresholds.
                Default ranges (based on base slat_steps=20):
                    - < 8,000 voxels: 1.5x steps (30 steps) - very simple objects
                    - 8,000 - 15,000: 1.3x steps (26 steps) - small objects  
                    - 15,000 - 30,000: 1.15x steps (23 steps) - medium complexity
                    - 30,000 - 50,000: 1.0x steps (20 steps) - standard quality
                    - > 50,000: 0.85x steps (17 steps) - large objects
            cfg_ranges (List[tuple[int, float]]): List of (voxel_threshold, cfg_strength) tuples.
                Controls slat_cfg_strength dynamically based on voxel count.
                Default ranges (based on base slat_cfg_strength=2.4):
                    - < 8,000 voxels: 3.2 - stronger guidance for simple objects
                    - 8,000 - 15,000: 2.8 - slightly stronger
                    - 15,000 - 30,000: 2.4 - base value
                    - 30,000 - 50,000: 2.0 - softer for complex objects
                    - > 50,000: 1.7 - minimal guidance for very complex objects
            
        Returns:
            tuple: (outputs dict, number of occupied voxels, adjusted slat steps, adjusted cfg strength)
        """
        # Default voxel ranges with step multipliers (conservative around base=20)
        if voxel_ranges is None and dynamic_steps:
            voxel_ranges = [
                (8000, 1.5),     # Very small objects - 30 steps
                (15000, 1.3),    # Small objects - 26 steps
                (30000, 1.15),   # Medium complexity - 23 steps
                (50000, 1.0),    # Large objects - 20 steps (base)
                (80000, 0.85),   # Very large objects - 17 steps
            ]
        
        # Default CFG ranges (conservative around base=2.4)
        if cfg_ranges is None and dynamic_steps:
            cfg_ranges = [
                (8000, 3.2),     # Very small - strong guidance
                (15000, 2.8),    # Small - slightly stronger
                (30000, 2.4),    # Medium - base value
                (50000, 2.0),    # Large - softer guidance
                (80000, 1.7),    # Very large - minimal guidance
            ]
        
        if preprocess_image:
            images = [self.preprocess_image(image) for image in images]
        cond = self.get_cond(images)
        cond["neg_cond"] = cond["neg_cond"][:1]
        torch.manual_seed(seed)
        num_oversamples = max(num_samples, num_oversamples)
        ss_steps = {
            **self.sparse_structure_sampler_params,
            **sparse_structure_sampler_params,
        }.get("steps")
        with self.inject_sampler_multi_image(
            "sparse_structure_sampler", len(images), ss_steps, mode=mode
        ):
            coords = self.sample_sparse_structure(
                cond, num_oversamples, sparse_structure_sampler_params
            )
            coords = (
                coords
                if num_oversamples <= num_samples
                else self.select_coords(coords, num_samples)
            )
        
        # Count occupied voxels
        num_voxels = len(coords)
        
        # Calculate step multiplier based on voxel count with interpolation
        base_slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get("steps")
        if dynamic_steps:
            step_multiplier = self._interpolate_value(num_voxels, voxel_ranges)
        else:
            step_multiplier = 1.0

        adjusted_slat_steps = int(base_slat_steps * step_multiplier)
        
        # Calculate CFG strength based on voxel count with interpolation
        if dynamic_steps:
            adjusted_cfg_strength = self._interpolate_value(num_voxels, cfg_ranges)
        else:
            adjusted_cfg_strength = {**self.slat_sampler_params, **slat_sampler_params}.get("cfg_strength")
        
        # Update slat_sampler_params with adjusted steps and cfg_strength
        adjusted_slat_sampler_params = {
            **slat_sampler_params, 
            "steps": adjusted_slat_steps,
            "cfg_strength": adjusted_cfg_strength,
        }
        
        with self.inject_sampler_multi_image(
            "slat_sampler", len(images), adjusted_slat_steps, mode=mode
        ):
            slat = self.sample_slat(cond, coords, adjusted_slat_sampler_params)
        
        outputs = self.decode_slat(slat, formats)
        return outputs, num_voxels, adjusted_slat_steps, adjusted_cfg_strength
