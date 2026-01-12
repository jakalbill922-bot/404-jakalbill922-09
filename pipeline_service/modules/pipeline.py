from __future__ import annotations

import base64
import io
import time
from typing import Literal, Optional

from PIL import Image
import torch
import gc

from config import Settings, settings
from logger_config import logger
from schemas import (
    GenerateRequest,
    GenerateResponse,
    TrellisParams,
    TrellisRequest,
    TrellisResult,
)
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.background_removal.rmbg_manager import BackgroundRemovalService
from modules.gs_generator.trellis_manager import TrellisService
from modules.utils import (
    secure_randint,
    set_random_seed,
    decode_image,
    to_png_base64,
    save_files,
)

import cv2
import numpy as np
from pathlib import Path

class GenerationPipeline:
    def __init__(self, settings: Settings = settings):
        self.settings = settings
        self.qwen_edit = QwenEditModule(settings)
        self.rmbg = BackgroundRemovalService(settings)
        self.trellis = TrellisService(settings)

    async def startup(self) -> None:
        logger.info("Starting pipeline")
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        await self.qwen_edit.startup()
        await self.rmbg.startup()
        await self.trellis.startup()
        # logger.info("Warming up generator...")
        # await self.warmup_generator()
        self._clean_gpu_memory()
        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        logger.info("Closing pipeline")
        await self.qwen_edit.shutdown()
        await self.rmbg.shutdown()
        await self.trellis.shutdown()
        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()
        
        
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

    # --- HÀM CỐT LÕI 1: CHUẨN BỊ ẢNH (CHỈ CHẠY 1 LẦN) ---
    async def prepare_input_images(
        self, image_bytes: bytes, seed: int = 42
    ) -> tuple[Image.Image, Image.Image]:
        """Chạy Qwen và RMBG để tạo view. Tách rời để dùng lại cho nhiều seed Trellis."""
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image = decode_image(image_base64)
        if seed < 0:
            seed = secure_randint(0, 10000)
        set_random_seed(seed)

        is_special_image = await self.select_feature(image)

        if is_special_image:
            logger.info("Special image detected, adjusting processing accordingly.")
             # 1. left view
            left_image_edited = self.qwen_edit.edit_image(
                prompt_image=image,
                seed=seed,
                prompt="Show this object in left three-quarters view and make sure it is fully visible. Turn background black color. Persist near object on background. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
            )

            # right view
            right_image_edited = self.qwen_edit.edit_image(
                prompt_image=image,
                seed=seed,
                prompt="Show this object in right three-quarters view and make sure it is fully visible. Turn background black color. Persist near object on background. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
            )

            # back view
            back_image_edited = self.qwen_edit.edit_image(
                prompt_image=image,
                seed=seed,
                prompt="Show this object in back three-quarters view and make sure it is fully visible. Turn background black color. Persist near object on background. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
            )

            return [
                left_image_edited,
                right_image_edited,
                back_image_edited,
            ]
        else:
            # 1. left view
            left_image_edited = self.qwen_edit.edit_image(
                prompt_image=image,
                seed=seed,
                prompt="Show this object in left three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
            )

            # right view
            right_image_edited = self.qwen_edit.edit_image(
                prompt_image=image,
                seed=seed,
                prompt="Show this object in right three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
            )

            # back view
            # back_image_edited = self.qwen_edit.edit_image(
            #     prompt_image=image,
            #     seed=seed,
            #     prompt="Show this object in back three-quarters view and make sure it is fully visible. Turn background neutral solid color contrasting with an object. Delete background details. Delete watermarks. Keep object colors. Sharpen image details",
            # )

            # 2. Remove background
            left_image_without_background = self.rmbg.remove_background(left_image_edited)
            right_image_without_background = self.rmbg.remove_background(right_image_edited)
            # back_image_without_background = self.rmbg.remove_background(back_image_edited)
            original_image_without_background = self.rmbg.remove_background(image)

            return [
                left_image_without_background,
                right_image_without_background,
                # back_image_without_background,
                original_image_without_background,
            ]

    # --- HÀM CỐT LÕI 2: CHẠY TRELLIS (CHẠY NHIỀU LẦN VỚI SEED KHÁC NHAU) ---
    async def generate_trellis_only(
        self,
        processed_images: list[Image.Image],
        seed: int,
        mode: Literal[
            "single", "multi_multi", "multi_sto", "multi_with_voxel_count"
        ] = "multi_with_voxel_count",
    ) -> bytes:
        """Chỉ chạy tạo 3D từ ảnh đã xử lý."""
        trellis_params = TrellisParams.from_settings(self.settings)
        set_random_seed(seed)

        trellis_result = self.trellis.generate(
            TrellisRequest(
                images=processed_images,
                seed=seed,
                params=trellis_params,
            ),
            mode=mode,
        )

        if not trellis_result or not trellis_result.ply_file:
            raise ValueError("Trellis generation failed")

        return trellis_result.ply_file

    # --- API Wrapper Cũ (Refactored) ---
    async def generate_gs(self, request: GenerateRequest) -> GenerateResponse:
        t1 = time.time()
        logger.info(f"New generation request")

        if request.seed < 0:
            request.seed = secure_randint(0, 10000)

        # Decode từ request để lấy bytes cho hàm prepare
        img_bytes = base64.b64decode(request.prompt_image)

        # 1. Prepare Images
        processed_image = await self.prepare_input_images(img_bytes, request.seed)

        # 2. Generate Trellis
        ply_bytes = await self.generate_trellis_only(processed_image, request.seed)

        # 3. Tạo kết quả trả về (Mock lại TrellisResult để save file nếu cần)
        # Lưu ý: Logic save file cũ đang nằm rải rác, mình giả lập lại response
        if self.settings.save_generated_files:
            # Reconstruct dummy result object if needed for saving logic
            pass

        t2 = time.time()
        generation_time = t2 - t1
        logger.info(f"Total generation time: {generation_time} seconds")
        self._clean_gpu_memory()

        return GenerateResponse(
            generation_time=generation_time,
            ply_file_base64=ply_bytes,  # Trả về bytes trực tiếp, controller sẽ encode base64
            # Các trường image_edited tạm thời để None hoặc cần logic riêng để lấy ra từ processed_imgs nếu muốn trả về
            image_edited_file_base64=to_png_base64(processed_image)
            if self.settings.send_generated_files
            else None,
            image_without_background_file_base64=to_png_base64(processed_image)
            if self.settings.send_generated_files
            else None,
        )

    # API cũ wrapper
    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        # Tái sử dụng logic mới
        processed_image = await self.prepare_input_images(image_bytes, seed)
        return await self.generate_trellis_only(processed_image, seed)
