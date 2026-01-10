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

        images = request.image if isinstance(request.image, Iterable) else [request.image]
        images_rgb = [image.convert("RGB") for image in images]
        num_images = len(images_rgb)
        
        logger.info(f"Generating Trellis {request.seed=} and image size {images[0].size} (Using {num_images} images)")

        params = self.default_params.overrided(request.params)

        start = time.time()
        buffer = None
        try:
            # Use different pipeline methods based on number of images
            if num_images == 1:
                # Single image mode - use standard run() method
                logger.info("Using single-image pipeline (run)")
                outputs = self.pipeline.run(
                    images_rgb[0],
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
                )
            else:
                # Multi-image mode - use run_multi_image() with mode
                logger.info(f"Using multi-image pipeline (run_multi_image) with mode={params.mode}")
                outputs = self.pipeline.run_multi_image(
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
                    mode=params.mode
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

