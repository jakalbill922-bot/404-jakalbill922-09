from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Literal, Optional
import io

import torch
from PIL import Image, ImageStat

from config import Settings
from logger_config import logger
from libs.trellis.pipelines import TrellisImageTo3DPipeline
from schemas import TrellisResult, TrellisRequest, TrellisParams

class TrellisService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline: Optional[TrellisImageTo3DPipeline] = None
        self.gpu = settings.trellis_gpu
        self.default_params = TrellisParams.from_settings(self.settings)

    async def startup(self) -> None:
        logger.info("Loading Trellis pipeline...")
        logger.warning("Downloading models from HuggingFace - this may take 5-15 minutes on first run (model files are several GB)")
        os.environ.setdefault("ATTN_BACKEND", "flash-attn")
        os.environ.setdefault("SPCONV_ALGO", "native")

        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu)

        start_time = time.time()
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
            self.settings.trellis_model_id
        )
        load_time = time.time() - start_time
        logger.info(f"Models loaded in {load_time:.2f}s. Moving to GPU...")
        
        self.pipeline.cuda()
        total_time = time.time() - start_time
        logger.success(f"Trellis pipeline ready. Total startup time: {total_time:.2f}s")

    async def shutdown(self) -> None:
        self.pipeline = None
        logger.info("Trellis pipeline closed.")

    def is_ready(self) -> bool:
        return self.pipeline is not None
    
    def generate_shapes_only(
        self,
        trellis_request: TrellisRequest,
        num_samples: int = 5,
    ) -> list[tuple[torch.Tensor, int]]:
        """
        Generate only sparse structures (shapes) for multiple candidates.
        Returns list of (coords, voxel_count) tuples.
        
        Args:
            trellis_request: Request with images and parameters
            num_samples: Number of shapes to generate
            
        Returns:
            List of (coords tensor, voxel count) tuples
        """
        if not self.pipeline:
            raise RuntimeError("Trellis pipeline not loaded.")

        images_rgb = [image.convert("RGB") for image in trellis_request.images]
        logger.info(f"🎲 Generating {num_samples} shape candidates (sparse structures only)...")

        params = self.default_params.overrided(trellis_request.params)
        cond = self.pipeline.get_cond(images_rgb)
        cond["neg_cond"] = cond["neg_cond"][:1]
        
        torch.manual_seed(trellis_request.seed)
        start = time.time()
        
        try:
            ss_steps = params.sparse_structure_steps
            with self.pipeline.inject_sampler_multi_image(
                "sparse_structure_sampler", len(images_rgb), ss_steps, mode="multidiffusion"
            ):
                coords = self.pipeline.sample_sparse_structure(
                    cond, 
                    num_samples,
                    {"steps": ss_steps, "cfg_strength": params.sparse_structure_cfg_strength}
                )
            
            # Split coords by sample and count voxels
            results = []
            for sample_idx in range(num_samples):
                sample_coords = coords[coords[:, 0] == sample_idx]
                voxel_count = len(sample_coords)
                results.append((sample_coords, voxel_count))
                logger.info(f"   Shape {sample_idx+1}: {voxel_count} voxels")
            
            generation_time = time.time() - start
            logger.success(f"✅ Generated {num_samples} shapes in {generation_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Shape generation failed: {e}")
            raise
    
    def generate_textures_for_shapes(
        self,
        trellis_request: TrellisRequest,
        shapes: list[tuple[torch.Tensor, int]],
    ) -> list[TrellisResult]:
        """
        Generate textures for multiple shapes in ONE batched pass.
        
        Args:
            trellis_request: Request with images and parameters
            shapes: List of (coords, voxel_count) tuples
            
        Returns:
            List of TrellisResult with PLY files
        """
        if not self.pipeline:
            raise RuntimeError("Trellis pipeline not loaded.")

        images_rgb = [image.convert("RGB") for image in trellis_request.images]
        logger.info(f"🎨 Batch generating {len(shapes)} textures (one per shape)...")
        for idx, (_, voxel_count) in enumerate(shapes):
            logger.info(f"   Shape {idx+1}: {voxel_count} voxels")

        params = self.default_params.overrided(trellis_request.params)
        cond = self.pipeline.get_cond(images_rgb)
        cond["neg_cond"] = cond["neg_cond"][:1]
        
        # Use max voxel count to determine steps (conservative approach)
        max_voxel_count = max(voxel_count for _, voxel_count in shapes)
        base_slat_steps = params.slat_steps
        voxel_threshold = 25000
        
        if max_voxel_count > voxel_threshold:
            adjusted_slat_steps = base_slat_steps
            logger.info(f"Max voxel count {max_voxel_count} > {voxel_threshold}: Using standard texture steps ({adjusted_slat_steps})")
        else:
            adjusted_slat_steps = int(base_slat_steps * 1.5)
            logger.info(f"Max voxel count {max_voxel_count} <= {voxel_threshold}: Using increased texture steps ({adjusted_slat_steps})")
        
        # Combine all coords into one batch tensor
        # Re-index sample_idx for each shape
        combined_coords_list = []
        for new_sample_idx, (coords, _) in enumerate(shapes):
            # Update sample index (first column) to new batch index
            batch_coords = coords.clone()
            batch_coords[:, 0] = new_sample_idx
            combined_coords_list.append(batch_coords)
        
        combined_coords = torch.cat(combined_coords_list, dim=0)
        logger.info(f"📦 Combined {len(shapes)} shapes into batched coords: {combined_coords.shape}")
        
        start = time.time()
        
        try:
            # Set seed for batched generation
            torch.manual_seed(trellis_request.seed + 100)
            
            # Generate ALL textures in ONE batched pass
            with self.pipeline.inject_sampler_multi_image(
                "slat_sampler", len(images_rgb), adjusted_slat_steps, mode="multidiffusion"
            ):
                slat = self.pipeline.sample_slat(
                    cond, 
                    combined_coords,
                    {"steps": adjusted_slat_steps, "cfg_strength": params.slat_cfg_strength}
                )
            
            # Decode to Gaussians (batch output)
            outputs = self.pipeline.decode_slat(slat, formats=["gaussian"])
            gaussians = outputs["gaussian"]  # List of Gaussian models
            
            # Convert each to PLY
            results = []
            for i, gaussian in enumerate(gaussians):
                buffer = io.BytesIO()
                gaussian.save_ply(buffer)
                buffer.seek(0)
                result = TrellisResult(ply_file=buffer.getvalue())
                buffer.close()
                results.append(result)
            
            generation_time = time.time() - start
            logger.success(f"✅ Batch generated {len(shapes)} textures in {generation_time:.2f}s ({generation_time/len(shapes):.2f}s per texture)")
            
            return results
            
        except Exception as e:
            logger.error(f"Texture generation failed: {e}")
            raise

    def generate_batch(
        self,
        trellis_request: TrellisRequest,
        num_samples: int = 2,
        mode: Literal["single", "multi_sto", "multi_with_voxel_count"] = "multi_sto",
    ) -> list[TrellisResult]:
        """
        Generate multiple samples in one batched pass (same mode only).
        30-40% faster than sequential generation.
        
        Args:
            trellis_request: Request with images and parameters
            num_samples: Number of models to generate in batch
            mode: Generation mode
            
        Returns:
            List of TrellisResult, one per sample
        """
        if not self.pipeline:
            raise RuntimeError("Trellis pipeline not loaded.")

        images_rgb = [image.convert("RGB") for image in trellis_request.images]
        logger.info(f"Batch generating {num_samples} samples with mode={mode}, seed={trellis_request.seed}")

        params = self.default_params.overrided(trellis_request.params)
        start = time.time()
        
        try:
            # Generate multiple samples in ONE pass
            if mode == "single":
                outputs = self.pipeline.run(
                    image=images_rgb[0],
                    num_samples=num_samples,
                    seed=trellis_request.seed,
                    sparse_structure_sampler_params={
                        "steps": params.sparse_structure_steps,
                        "cfg_strength": params.sparse_structure_cfg_strength,
                    },
                    slat_sampler_params={
                        "steps": params.slat_steps,
                        "cfg_strength": params.slat_cfg_strength,
                    },
                    preprocess_image=False,
                    formats=["gaussian"],
                    num_oversamples=num_samples,
                )
            elif mode == "multi_sto":
                outputs = self.pipeline.run_multi_image(
                    images=images_rgb,
                    num_samples=num_samples,
                    seed=trellis_request.seed,
                    sparse_structure_sampler_params={
                        "steps": params.sparse_structure_steps,
                        "cfg_strength": params.sparse_structure_cfg_strength,
                    },
                    slat_sampler_params={
                        "steps": params.slat_steps,
                        "cfg_strength": params.slat_cfg_strength,
                    },
                    preprocess_image=False,
                    formats=["gaussian"],
                    num_oversamples=num_samples,
                    mode="stochastic",
                )
            elif mode == "multi_with_voxel_count":
                outputs, num_voxels = self.pipeline.run_multi_image_with_voxel_count(
                    images_rgb,
                    num_samples=num_samples,
                    seed=trellis_request.seed,
                    sparse_structure_sampler_params={
                        "steps": params.sparse_structure_steps,
                        "cfg_strength": params.sparse_structure_cfg_strength,
                    },
                    slat_sampler_params={
                        "steps": params.slat_steps,
                        "cfg_strength": params.slat_cfg_strength,
                    },
                    preprocess_image=False,
                    formats=["gaussian"],
                    num_oversamples=num_samples,
                    voxel_threshold=25000,
                )
            else:
                raise ValueError(f"Unsupported mode: {mode}")
                
            generation_time = time.time() - start
            
            # outputs["gaussian"] is a list of Gaussian models
            gaussians = outputs["gaussian"]
            
            # Convert each to PLY bytes
            results = []
            for i, gaussian in enumerate(gaussians):
                buffer = io.BytesIO()
                gaussian.save_ply(buffer)
                buffer.seek(0)
                result = TrellisResult(ply_file=buffer.getvalue())
                results.append(result)
                buffer.close()
            
            logger.success(
                f"Batch generation completed: {num_samples} samples in {generation_time:.2f}s "
                f"({generation_time/num_samples:.2f}s per sample)"
            )
            return results
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            raise

    def generate_dual_mode(
        self,
        images_voxel: list,
        images_sto: list,
        seed: int,
        params: TrellisParams,
    ) -> tuple[TrellisResult, TrellisResult]:
        """
        Generate both voxel_count and stochastic modes efficiently by sharing preprocessing.
        
        Returns:
            (result_voxel_count, result_stochastic)
        """
        if not self.pipeline:
            raise RuntimeError("Trellis pipeline not loaded.")
            
        logger.info(f"Dual-mode generation with seed={seed}")
        
        # Preprocess images
        images_voxel_rgb = [img.convert("RGB") for img in images_voxel]
        images_sto_rgb = [img.convert("RGB") for img in images_sto]
        
        start = time.time()
        
        try:
            # Generate model 1: multi_with_voxel_count
            outputs1, _ = self.pipeline.run_multi_image_with_voxel_count(
                images_voxel_rgb,
                seed=seed,
                sparse_structure_sampler_params={
                    "steps": params.sparse_structure_steps,
                    "cfg_strength": params.sparse_structure_cfg_strength,
                },
                slat_sampler_params={
                    "steps": params.slat_steps,
                    "cfg_strength": params.slat_cfg_strength,
                },
                preprocess_image=False,
                formats=["gaussian"],
                num_oversamples=params.num_oversamples,
                voxel_threshold=25000,
            )
            
            # Generate model 2: multi_sto (reuses GPU state)
            outputs2 = self.pipeline.run_multi_image(
                images_sto_rgb,
                seed=seed,
                sparse_structure_sampler_params={
                    "steps": params.sparse_structure_steps,
                    "cfg_strength": params.sparse_structure_cfg_strength,
                },
                slat_sampler_params={
                    "steps": params.slat_steps,
                    "cfg_strength": params.slat_cfg_strength,
                },
                preprocess_image=False,
                formats=["gaussian"],
                num_oversamples=params.num_oversamples,
                mode="stochastic",
            )
            
            generation_time = time.time() - start
            
            # Convert both to PLY bytes
            buffer1 = io.BytesIO()
            outputs1["gaussian"][0].save_ply(buffer1)
            buffer1.seek(0)
            
            buffer2 = io.BytesIO()
            outputs2["gaussian"][0].save_ply(buffer2)
            buffer2.seek(0)
            
            result1 = TrellisResult(ply_file=buffer1.getvalue())
            result2 = TrellisResult(ply_file=buffer2.getvalue())
            
            logger.success(f"Dual-mode generation completed in {generation_time:.2f}s")
            
            return result1, result2
            
        finally:
            if 'buffer1' in locals():
                buffer1.close()
            if 'buffer2' in locals():
                buffer2.close()

    def generate(
        self,
        trellis_request: TrellisRequest,
        mode: Literal["single", "multi_multi", "multi_sto", "multi_with_voxel_count"] = "single",
    ) -> TrellisResult:
        if not self.pipeline:
            raise RuntimeError("Trellis pipeline not loaded.")

        images_rgb = [image.convert("RGB") for image in trellis_request.images]
        logger.info(f"Generating Trellis {trellis_request.seed=} and image size {trellis_request.images[0].size}")

        params = self.default_params.overrided(trellis_request.params)

        start = time.time()
        try:
            if mode == "single":
                outputs = self.pipeline.run(
                    image=images_rgb[0],
                    seed=trellis_request.seed,
                    sparse_structure_sampler_params={
                        "steps": params.sparse_structure_steps,
                        "cfg_strength": params.sparse_structure_cfg_strength,
                    },
                    slat_sampler_params={
                        "steps": params.slat_steps,
                        "cfg_strength": params.slat_cfg_strength,
                        
                    },
                    preprocess_image=False,
                    formats=["gaussian"],
                    num_oversamples=params.num_oversamples,
                )
            elif mode == "multi_sto":
                outputs = self.pipeline.run_multi_image(
                    images=images_rgb,
                    seed=trellis_request.seed,
                    sparse_structure_sampler_params={
                        "steps": params.sparse_structure_steps,
                        "cfg_strength": params.sparse_structure_cfg_strength,
                    },
                    slat_sampler_params={
                        "steps": params.slat_steps,
                        "cfg_strength": params.slat_cfg_strength,
                        
                    },
                    preprocess_image=False,
                    formats=["gaussian"],
                    num_oversamples=params.num_oversamples,
                    mode="stochastic",
                )
            elif mode == "multi_with_voxel_count":
                outputs, num_voxels = self.pipeline.run_multi_image_with_voxel_count(
                    images_rgb,
                    seed=trellis_request.seed,
                    sparse_structure_sampler_params={
                        "steps": params.sparse_structure_steps,
                        "cfg_strength": params.sparse_structure_cfg_strength,
                    },
                    slat_sampler_params={
                        "steps": params.slat_steps,
                        "cfg_strength": params.slat_cfg_strength,
                    },
                    preprocess_image=False,
                    formats=["gaussian"],
                    num_oversamples=params.num_oversamples,
                    voxel_threshold=25000,
                )
            generation_time = time.time() - start
            gaussian = outputs["gaussian"][0]

            # Save ply to buffer
            buffer = io.BytesIO()
            gaussian.save_ply(buffer)
            buffer.seek(0)

            result = TrellisResult(
                ply_file=buffer.getvalue() if buffer else None # bytes
            )

            logger.success(f"Trellis finished generation in {generation_time:.2f}s.")
            return result
        finally:
            if buffer:
                buffer.close()

