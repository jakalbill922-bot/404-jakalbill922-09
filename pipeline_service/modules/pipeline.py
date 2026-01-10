from __future__ import annotations

import base64
import io
import time
from datetime import datetime
from typing import Optional

from PIL import Image
import pyspz
import torch
import gc

from config.settings import settings, SettingsConf
from config.prompting_library import PromptingLibrary
from logger_config import logger
from schemas import GenerateRequest, GenerateResponse, TrellisParams, TrellisRequest, TrellisResult
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.background_removal.ben2_module import BEN2BackgroundRemovalService
from modules.background_removal.rmbg20_module import RMBG2BackgroundRemovalService
from modules.gs_generator.trellis_manager import TrellisService
from modules.utils import image_grid, secure_randint, set_random_seed, decode_image, to_png_base64, save_files


class GenerationPipeline:
    """
    Generation pipeline 
    """

    def __init__(self, settings: SettingsConf = settings):
        self.settings = settings

        # Initialize modules
        self.qwen_edit = QwenEditModule(settings.qwen)

        # Initialize background removal module
        if self.settings.background_removal.model_id == "PramaLLC/BEN2":
            self.rmbg = BEN2BackgroundRemovalService(settings.background_removal)
        else:
            self.rmbg = RMBG2BackgroundRemovalService(settings.background_removal)

        # Initialize prompting libraries for both modes
        self.prompting_library_base = PromptingLibrary.from_file(settings.qwen.prompt_path_base)
        self.prompting_library_multistage = PromptingLibrary.from_file(settings.qwen.prompt_path_multistage)

        # Initialize Trellis module
        self.trellis = TrellisService(settings.trellis)

    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_edit.startup()
        await self.rmbg.startup()
        await self.trellis.startup()
        
        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()
        
        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        # Shutdown all modules
        await self.qwen_edit.shutdown()
        await self.rmbg.shutdown()
        await self.trellis.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """
        Clean the GPU memory.
        """
        gc.collect()
        torch.cuda.empty_cache()

    async def warmup_generator(self) -> None:
        """Function for warming up the generator"""
        
        temp_image = Image.new("RGB",(64,64),color=(128,128,128))
        buffer = io.BytesIO()
        temp_image.save(buffer, format="PNG")
        temp_imge_bytes = buffer.getvalue()
        await self.generate_from_upload(temp_imge_bytes,seed=42)

    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        """
        Generate 3D model from uploaded image file and return PLY as bytes.
        
        Args:
            image_bytes: Raw image bytes from uploaded file
            
        Returns:
            PLY file as bytes
        """
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        # Create request
        request = GenerateRequest(
            prompt_image=image_base64,
            prompt_type="image",
            seed=seed
        )
        
        # Generate
        response = await self.generate_gs(request)
        
        # Return binary PLY
        if not response.ply_file_base64:
            raise ValueError("PLY generation failed")
            
        return response.ply_file_base64 # bytes

    async def generate_gs(self, request: GenerateRequest) -> GenerateResponse:
        """
        Execute full generation pipeline.
        
        Args:
            request: Generation request with prompt and settings
            
        Returns:
            GenerateResponse with generated assets
        """
        t1 = time.time()
        logger.info(f"New generation request")

        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
            set_random_seed(request.seed)
        else:
            set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)

        # 1. Edit the image using Qwen Edit
        if self.settings.trellis.multiview:
            # Multiview mode: generate multiple views (3 images total)
            logger.info("Multiview mode: generating multiple views")
            background_prompt = self.prompting_library_multistage.promptings['background']
            views_prompt = self.prompting_library_multistage.promptings['views']
            
            # Stage 0: Remove background
            images_edited = self.qwen_edit.edit_image(
                prompt_image=image, 
                seed=request.seed, 
                prompting=background_prompt, 
                encode_prompt=False
            )
            # Stage 1: Generate novel views (2 additional views)
            images_edited += self.qwen_edit.edit_image(
                prompt_image=images_edited[0], 
                seed=request.seed, 
                prompting=views_prompt
            )
        else:
            # Base mode: only clean background, single view (1 image)
            logger.info("Base mode: single view with background cleaning and rotation")
            base_prompt = self.prompting_library_base.promptings['base']
            images_edited = self.qwen_edit.edit_image(
                prompt_image=image, 
                seed=request.seed, 
                prompting=base_prompt
            )

        # 2. Remove background
        images_with_background = list(image.copy() for image in images_edited)
        images_without_background = self.rmbg.remove_background(images_with_background)

        trellis_result: Optional[TrellisResult] = None
        
        # Resolve Trellis parameters from request
        trellis_params: TrellisParams = request.trellis_params
       
        # 3. Generate the 3D model
        trellis_result = self.trellis.generate(
            TrellisRequest(
                image=images_without_background,
                seed=request.seed,
                params=trellis_params
            )
        )

        image_edited_base64 = None
        image_without_background_base64 = None

        if self.settings.output.save_generated_files or self.settings.output.send_generated_files:
            image_edited = image_grid(images_edited)
            image_without_background = image_grid(images_without_background)

            # Save generated files
            if self.settings.output.save_generated_files:
                save_files(trellis_result, image_edited, image_without_background)
            
            # Convert to PNG base64 for response (only if needed)
            if self.settings.output.send_generated_files:
                
                image_edited_base64 = to_png_base64(image_edited)
                image_without_background_base64 = to_png_base64(image_without_background)

        t2 = time.time()
        generation_time = t2 - t1

        logger.success(f"Generation time: {generation_time:.2f}s")

        # Clean the GPU memory
        self._clean_gpu_memory()

        response = GenerateResponse(
            generation_time=generation_time,
            ply_file_base64=trellis_result.ply_file if trellis_result else None,
            image_edited_file_base64=image_edited_base64 if self.settings.output.send_generated_files else None,
            image_without_background_file_base64=image_without_background_base64 if self.settings.output.send_generated_files else None,
        )
        return response

