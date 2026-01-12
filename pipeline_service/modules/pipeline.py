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
from modules.background_removal.birefnet_module import BirefNetBackgroundRemovalService
from modules.gs_generator.trellis_manager import TrellisService
from modules.utils import image_grid, secure_randint, set_random_seed, decode_image, to_png_base64, save_files

import cv2
import numpy as np
from pathlib import Path

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
        elif self.settings.background_removal.model_id == "ZhengPeng7/BiRefNet_dynamic":
            self.rmbg = BirefNetBackgroundRemovalService(settings.background_removal)
        else:
            raise ValueError(f"Unsupported background removal model: {self.settings.background_removal.model_id}")

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

    def _edit_images(self, image: Image.Image, seed: int) -> list[Image.Image]:
        """
        Edit image based on current mode (multiview or base).
        
        Args:
            image: Input image to edit
            seed: Random seed for reproducibility
            
        Returns:
            List of edited images
        """
        if self.settings.trellis.multiview:
            logger.info("Multiview mode: generating multiple views")
            background_prompt = self.prompting_library_multistage.promptings['background']
            views_prompt = self.prompting_library_multistage.promptings['views']
            # print(f"Background Prompt: {background_prompt.prompt}")
            # print(f"Views Prompt: {views_prompt.prompt}")
            # Stage 0: Remove background
            images = self.qwen_edit.edit_image(
                prompt_image=image,
                seed=seed,
                prompting=background_prompt,
                encode_prompt=False
            )
            # Stage 1: Generate novel views (2 additional views)
            images += self.qwen_edit.edit_image(
                prompt_image=image,
                seed=seed,
                prompting=views_prompt,
                encode_prompt=True
            )
            return images
        
        # Base mode: only clean background, single view (1 image)
        logger.info("Base mode: single view with background cleaning and rotation")
        base_prompt = self.prompting_library_base.promptings['base']
        return self.qwen_edit.edit_image(
            prompt_image=image,
            seed=seed,
            prompting=base_prompt
        )

    def _prepare_output(
        self,
        images_edited: list[Image.Image],
        images_without_background: list[Image.Image],
        trellis_result: TrellisResult
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Prepare output files (save and/or encode to base64).
        
        Args:
            images_edited: List of edited images
            images_without_background: List of images with removed background
            trellis_result: Result from Trellis generation
            
        Returns:
            Tuple of (image_edited_base64, image_without_background_base64)
        """
        if not (self.settings.output.save_generated_files or self.settings.output.send_generated_files):
            return None, None
        
        image_edited_grid = image_grid(images_edited)
        image_no_bg_grid = image_grid(images_without_background)
        
        if self.settings.output.save_generated_files:
            save_files(trellis_result, image_edited_grid, image_no_bg_grid)
        
        if self.settings.output.send_generated_files:
            return to_png_base64(image_edited_grid), to_png_base64(image_no_bg_grid)
        
        return None, None


    async def select_feature(self, image: Image.Image, tolerance: float = 1e-5) -> bool:
        """
        Select feature based on input image.
        """
        img_array = np.array(image)
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        # img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # brightness & contrast
        brightness = float(gray.mean())
        contrast = float(gray.std())

        # edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = edges.astype(bool).mean()

        # entropy (rough texture measure)
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        p = hist / hist.sum()
        p = p[p > 0]
        entropy = float(-(p * np.log2(p)).sum())
        
        # Create input tensor
        input_values = torch.tensor([brightness, contrast, edge_density, entropy], dtype=torch.float32)
        logger.info(f"Extracted features: brightness={brightness}, contrast={contrast}, edge_density={edge_density}, entropy={entropy}")
        # Check if exists (with tolerance for floating point comparison)
        is_special_image = False
        
        if (brightness > 85 and brightness < 191) and (contrast > 34 and contrast < 53) and (edge_density > 0.011 and edge_density < 0.054) and (entropy > 5.9 and entropy < 7.5):
            is_special_image = True
            logger.info("Image classified as special based on feature thresholds.")
        
        return is_special_image

    async def generate_gs(self, request: GenerateRequest) -> GenerateResponse:
        """
        Execute full generation pipeline.
        
        Args:
            request: Generation request with prompt and settings
            
        Returns:
            GenerateResponse with generated assets
        """
        t1 = time.time()
        logger.info("New generation request")

        try:
            # Set seed
            if request.seed < 0:
                request.seed = secure_randint(0, 10000)
            set_random_seed(request.seed)

            # Decode input image
            image = decode_image(request.prompt_image)

            is_special_image = await self.select_feature(image)

            # 1. Edit the image using Qwen Edit
            images_edited = self._edit_images(image, request.seed)

            # 2. Remove background
            images_with_background = [img.copy() for img in images_edited]
            if is_special_image:
                images_without_background = images_with_background
            else:
                images_without_background = self.rmbg.remove_background(images_with_background)
            # images_without_background = self.rmbg.remove_background(images_with_background)
            
            
            print(f"Images edited: {len(images_edited)}, Images without background: {len(images_without_background)}")

            # 3. Generate the 3D model
            trellis_result = self.trellis.generate(
                TrellisRequest(
                    image=images_without_background,
                    seed=request.seed,
                    params=request.trellis_params
                )
            )

            # 4. Prepare output files
            image_edited_base64, image_no_bg_base64 = self._prepare_output(
                images_edited, images_without_background, trellis_result
            )

            generation_time = time.time() - t1
            logger.success(f"Generation time: {generation_time:.2f}s")

            return GenerateResponse(
                generation_time=generation_time,
                ply_file_base64=trellis_result.ply_file,
                image_edited_file_base64=image_edited_base64,
                image_without_background_file_base64=image_no_bg_base64,
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

        finally:
            self._clean_gpu_memory()

