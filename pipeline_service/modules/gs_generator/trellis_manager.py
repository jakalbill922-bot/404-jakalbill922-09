from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Iterable, Optional
import io

import torch
from PIL import Image, ImageStat

from config.settings import TrellisConfig
from logger_config import logger
from libs.trellis.pipelines import TrellisImageTo3DPipeline
from schemas import TrellisResult, TrellisRequest, TrellisParams

class TrellisService:
    def __init__(self, settings: TrellisConfig):
        self.settings = settings
        self.pipeline: Optional[TrellisImageTo3DPipeline] = None
        self.gpu = settings.gpu
        self.default_params = TrellisParams.from_settings(self.settings)

    async def startup(self) -> None:
        logger.info("Loading Trellis pipeline...")
        os.environ.setdefault("ATTN_BACKEND", "flash-attn")
        os.environ.setdefault("SPCONV_ALGO", "native")

        if torch.cuda.is_available():
            torch.cuda.set_device(self.gpu)

        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(
            self.settings.model_id
        )
        self.pipeline.cuda()
        logger.success("Trellis pipeline ready.")

    async def shutdown(self) -> None:
        self.pipeline = None
        logger.info("Trellis pipeline closed.")

    def is_ready(self) -> bool:
        return self.pipeline is not None

    def generate(
        self,
        request: TrellisRequest,
    ) -> TrellisResult:
        if not self.pipeline:
            raise RuntimeError("Trellis pipeline not loaded.")

        
        # Convert image to list if it's not already
        images = request.image if isinstance(request.image, Iterable) else [request.image]
        images_rgb = [image.convert("RGB") for image in images]
        num_images = len(images_rgb)
        
        # Override default parameters with user-provided parameters
        params = self.default_params.overrided(request.params)

        logger.info(f"Generating Trellis {request.seed=} and image size {images[0].size} | Using {num_images} images | Mode: {params.mode}")

        start = time.time()
        buffer = None
        try:
            outputs, num_voxels, adjusted_slat_steps, adjusted_cfg_strength = self.pipeline.run_multi_image_with_voxel_count(
                images_rgb,
                seed=request.seed,
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
                mode=params.mode,
                dynamic_steps=params.dynamic_steps,
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

            logger.success(f"Trellis finished generation in {generation_time:.2f}s. | Voxel count: {num_voxels} | Adjusted slat steps: {adjusted_slat_steps} | Adjusted cfg strength: {adjusted_cfg_strength}")
            return result
        finally:
            if buffer:
                buffer.close()

