from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import base64

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import pyspz
from plyfile import PlyData
import torch

from config import settings
from logger_config import logger
from schemas import GenerateRequest, GenerateResponse, TrellisRequest, TrellisParams
from modules import GenerationPipeline
from modules.duel_manager import DuelManager
from modules.utils import secure_randint, set_random_seed
from PIL import Image
import io

duel_manager = DuelManager(settings)
pipeline = GenerationPipeline(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await pipeline.startup()
    # Warmup disabled to save GPU memory - models are already loaded
    temp_image = Image.new("RGB", (64, 64), color=(128, 128, 128))
    buffer = io.BytesIO()
    temp_image.save(buffer, format="PNG")
    temp_imge_bytes = buffer.getvalue()
    await run_champion_generation(temp_imge_bytes, -1)
    pipeline._clean_gpu_memory()
    try:
        yield
    finally:
        await pipeline.shutdown()


app = FastAPI(
    title=settings.api_title,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def run_champion_generation(
    image_bytes: bytes, seed: int
) -> tuple[bytes, int, str]:
    """
    BATCHED MULTIDIFFUSION WITH AUTOMATIC VARIETY:
    1. Batch generate N candidates with multidiffusion mode (30-40% faster!)
    2. VLLM judges all candidates and picks the best
    
    Important: Even with SAME seed, batching produces DIFFERENT models because:
    - torch.randn(num_samples, ...) generates different noise per sample
    - Each sample gets sequential values from the RNG stream
    - Result: Natural variety without sacrificing speed!
    
    TIMEOUT: If judging takes > 25s, returns the current best candidate.
    """
    import asyncio
    import time
    
    final_ply_bytes = None

    if seed < 0:
        seed = secure_randint(0, 10000)
    set_random_seed(seed)

    # START TIMER: Begin when Qwen editing starts
    generation_start = time.time()
    
    # STAGE 0: Generate baseline PLY from raw image (no editing, no background removal)
    logger.info("🎯 STAGE 0: Generating baseline PLY from raw image...")
    raw_image = Image.open(BytesIO(image_bytes))
    from schemas import TrellisRequest, TrellisParams
    params = TrellisParams.from_settings(settings)
    baseline_trellis_req = TrellisRequest(
        images=[raw_image],
        seed=seed,
        params=params,
    )
    # Generate one shape and texture from raw image
    baseline_shape = pipeline.trellis.generate_shapes_only(baseline_trellis_req, num_samples=1)
    baseline_candidates = pipeline.trellis.generate_textures_for_shapes(
        baseline_trellis_req,
        baseline_shape
    )
    baseline_candidate = baseline_candidates[0]
    logger.success(f"✅ Baseline PLY generated from raw image")
    
    logger.info("⚡ Preprocessing images (One-time execution)...")
    
    (
        left_image_without_background,
        right_image_without_background,
        original_image_without_background,
    ) = await pipeline.prepare_input_images(image_bytes, seed)

    # STAGE 1: Generate max_candidates shape candidates
    logger.info(f"🎲 STAGE 1: Generating {settings.max_candidates} shape candidates...")
    trellis_req = TrellisRequest(
        images=[left_image_without_background, right_image_without_background, original_image_without_background],
        seed=seed,
        params=params,
    )
    
    shape_candidates = pipeline.trellis.generate_shapes_only(trellis_req, num_samples=settings.max_candidates)
    
    # Use first shape to determine number of texture candidates needed
    reference_voxel_count = shape_candidates[0][1]
    
    # Determine number of texture candidates based on voxel ranges
    num_texture_candidates = settings.candidate_ranges[-1][1]  # Default to last range's value
    for max_voxels, num_candidates in settings.candidate_ranges:
        if reference_voxel_count <= max_voxels:
            num_texture_candidates = num_candidates
            logger.info(f"📊 Voxel count {reference_voxel_count} ≤ {max_voxels} → {num_candidates} candidates")
            break
    else:
        logger.info(f"📊 Voxel count {reference_voxel_count} > {settings.candidate_ranges[-1][0]} → {num_texture_candidates} candidate(s)")
    
    # Determine timeout based on voxel count
    dynamic_timeout = settings.default_timeout  # Default to fallback
    for max_voxels, timeout_seconds in settings.timeout_ranges:
        if reference_voxel_count <= max_voxels:
            dynamic_timeout = timeout_seconds
            logger.info(f"⏱️ Voxel count {reference_voxel_count} ≤ {max_voxels} → {timeout_seconds}s timeout")
            break
    else:
        logger.info(f"⏱️ Voxel count {reference_voxel_count} > {settings.timeout_ranges[-1][0]} → {dynamic_timeout}s timeout (default)")
    
    # Select N smallest shapes from the 5 generated
    # Keep track of original indices during sorting
    indexed_shapes = [(i, coords, voxel_count) for i, (coords, voxel_count) in enumerate(shape_candidates)]
    sorted_indexed_shapes = sorted(indexed_shapes, key=lambda x: x[2])  # Sort by voxel count
    selected_indexed_shapes = sorted_indexed_shapes[:num_texture_candidates]  # Take N smallest
    
    logger.success(f"📐 Selected {num_texture_candidates} smallest shapes:")
    for original_idx, coords, voxel_count in selected_indexed_shapes:
        logger.info(f"   Shape {original_idx+1}: {voxel_count} voxels")
    
    # Extract just coords and voxel_count for texture generation
    selected_shapes = [(coords, voxel_count) for _, coords, voxel_count in selected_indexed_shapes]
    
    logger.info(f"🎨 STAGE 2: Generating {num_texture_candidates} textures (one per selected shape)")
    
    # STAGE 2: Generate textures for the selected smallest shapes
    candidates = pipeline.trellis.generate_textures_for_shapes(
        trellis_req,
        selected_shapes
    )
    
    # Insert baseline candidate at position 0
    candidates.insert(0, baseline_candidate)
    num_texture_candidates += 1  # Update count to include baseline
    logger.success(f"📋 Total candidates: {num_texture_candidates} (including baseline)")
    
    generation_time = time.time() - generation_start

    # OPTIMIZATION: Render all texture candidates ONCE upfront
    ply_files = [candidate.ply_file for candidate in candidates]
    rendered_images = await duel_manager.batch_render_candidates(ply_files)
    
    # Check if generation already took too long
    elapsed = time.time() - generation_start
    if elapsed > dynamic_timeout:
        logger.warning(f"⏱️ Generation took {elapsed:.2f}s > {dynamic_timeout}s timeout. Returning first candidate without judging.")
        return candidates[0].ply_file, seed
    
    # VLLM judges all candidates and selects the best with timeout
    logger.info(f"⚔️ VLLM judging {num_texture_candidates} candidates (Candidate 1 is baseline)...")
    
    best_idx = 0
    
    async def judge_candidates():
        nonlocal best_idx
        # Judge each candidate against the current best using PRE-RENDERED images
        for i in range(1, num_texture_candidates):
            candidate_label = "baseline" if best_idx == 0 else f"candidate {best_idx+1}"
            current_label = f"candidate {i+1}"
            logger.info(f"   -> Judging {current_label} vs {candidate_label}...")
            
            # Compare using pre-rendered images (no re-rendering!)
            winner_idx, issues = await duel_manager.run_duel_prerendered(
                image_bytes,
                rendered_images[best_idx],  # Baseline rendered image
                rendered_images[i]           # Current candidate rendered image
            )
            
            if winner_idx == 1:  # Current candidate won
                prev_label = "baseline" if best_idx == 0 else f"candidate {best_idx+1}"
                logger.info(f"   Candidate {i+1}: BETTER than {prev_label}")
                best_idx = i
            else:
                prev_label = "baseline" if best_idx == 0 else f"candidate {best_idx+1}"
                logger.info(f"   Candidate {i+1}: Worse than {prev_label}")
    
    # Run judging with timeout
    remaining_time = max(1, dynamic_timeout - elapsed)  # At least 1 second
    try:
        await asyncio.wait_for(judge_candidates(), timeout=remaining_time)
    except asyncio.TimeoutError:
        logger.warning(f"⏱️ Judging timeout after {remaining_time:.2f}s. Returning current best: Candidate {best_idx+1}")
    
    final_ply_bytes = candidates[best_idx].ply_file
    selected_seed = seed
    total_time = time.time() - generation_start
    champion_label = "Baseline (raw image)" if best_idx == 0 else f"Candidate {best_idx+1}"
    logger.success(f"🏆 CHAMPION: {champion_label} (total time: {total_time:.2f}s)")

    return final_ply_bytes, selected_seed


# ---------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ready"}


@app.post("/generate_from_base64", response_model=GenerateResponse)
async def generate_from_base64(request: GenerateRequest) -> GenerateResponse:
    """
    Endpoint JSON này giữ nguyên, thường dùng cho test single request.
    """
    try:
        # Lưu ý: generate_gs trong pipeline nên được cập nhật để dùng prepare_input_images bên trong
        result = await pipeline.generate_gs(request)

        compressed_ply_bytes = None
        if result.ply_file_base64 and settings.compression:
            compressed_ply_bytes = pyspz.compress(result.ply_file_base64, workers=1)
            logger.info(f"Compressed PLY size: {len(compressed_ply_bytes)} bytes")

        result.ply_file_base64 = base64.b64encode(
            result.ply_file_base64 if not compressed_ply_bytes else compressed_ply_bytes
        ).decode("utf-8")

        return result
    except Exception as exc:
        logger.exception(f"Error generating task: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate")
async def generate(
    prompt_image_file: UploadFile = File(...), seed: int = Form(-1)
) -> StreamingResponse:
    """
    Generate PLY (Streaming). Có chạy giải đấu nếu seed = -1.
    """
    try:
        logger.info(
            f"Task received (/generate). Uploading image: {prompt_image_file.filename}"
        )
        image_bytes = await prompt_image_file.read()

        # Gọi hàm chung
        final_ply_bytes, selected_seed = await run_champion_generation(
            image_bytes, seed
        )

        ply_buffer = BytesIO(final_ply_bytes)
        buffer_size = len(ply_buffer.getvalue())
        ply_buffer.seek(0)

        headers = {
            "Content-Length": str(buffer_size),
            "X-Generated-Seed": str(selected_seed),
        }

        async def generate_chunks():
            chunk_size = 1024 * 1024
            while chunk := ply_buffer.read(chunk_size):
                yield chunk

        return StreamingResponse(
            generate_chunks(), media_type="application/octet-stream", headers=headers
        )

    except Exception as exc:
        logger.exception(f"Error generating from upload: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate-spz")
async def generate_spz(
    prompt_image_file: UploadFile = File(...), seed: int = Form(-1)
) -> StreamingResponse:
    """
    Generate SPZ (Compressed). BẮT BUỘC CHẠY GIẢI ĐẤU nếu seed = -1 để lấy model tốt nhất.
    """
    try:
        logger.info(
            f"Task received (/generate-spz). Uploading image: {prompt_image_file.filename}"
        )
        image_bytes = await prompt_image_file.read()

        # 1. Gọi hàm chung để lấy model VÔ ĐỊCH (Champion)
        final_ply_bytes, selected_seed = await run_champion_generation(
            image_bytes, seed
        )

        # 2. Nén thành SPZ
        if final_ply_bytes:
            logger.info("Compressing Champion model to SPZ...")
            ply_compressed_bytes = pyspz.compress(final_ply_bytes, workers=1)
            logger.info(f"Task completed. SPZ size: {len(ply_compressed_bytes)} bytes")

            # (Tùy chọn) Có thể trả về Header thông tin người thắng cuộc
            headers = {
                "X-Generated-Seed": str(selected_seed),
            }

            return StreamingResponse(
                BytesIO(ply_compressed_bytes),
                media_type="application/octet-stream",
                headers=headers,
            )
        else:
            raise HTTPException(status_code=500, detail="Generated content is empty")

    except Exception as exc:
        logger.exception(f"Error generating from upload: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/setup/info")
async def get_setup_info() -> dict:
    try:
        return settings.dict()
    except Exception as e:
        logger.error(f"Failed to get setup info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host=settings.host, port=settings.port, reload=False)
