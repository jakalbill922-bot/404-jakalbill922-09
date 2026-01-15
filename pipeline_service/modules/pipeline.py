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
