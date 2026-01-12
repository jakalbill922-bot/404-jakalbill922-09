import math
from typing import Iterable, Optional
import torch
from diffusers import QwenImageEditPlusPipeline
import time
from PIL import Image

import json

from dotenv import load_dotenv

load_dotenv()

from logger_config import logger
import hashlib

from diffusers.models import QwenImageTransformer2DModel
from modules.image_edit.qwen_manager import QwenManager
from modules.image_edit.prompting import Prompting, TextPrompting, EmbeddedPrompting
from config.settings import QwenConfig

CONDITION_IMAGE_SIZE = 384 * 384
INPUT_IMAGE_SIZE = 1024 * 1024

class QwenEditModule(QwenManager):
    """Qwen module for image editing operations."""
    CONDITION_IMAGE_SIZE = 384 * 384
    INPUT_IMAGE_SIZE = 1024 * 1024

    def __init__(self, settings: QwenConfig):
        super().__init__(settings)
        self._empty_image = self._prepare_input_image(Image.new('RGB', (64, 64)))

        self.base_model_path = settings.base_model_path
        self.edit_model_path = settings.model_path

        self.pipe_config = {
            "num_inference_steps": settings.num_inference_steps,
            "true_cfg_scale": settings.true_cfg_scale,
            "height": settings.height,
            "width": settings.width,

        }


    def _get_model_transformer(self):
        """Load the Nunchaku Qwen transformer for image editing."""
        return  QwenImageTransformer2DModel.from_pretrained(
                self.edit_model_path,
                subfolder="transformer",
                torch_dtype=self.dtype
            )

    def _get_model_pipe(self, transformer, scheduler):

        return QwenImageEditPlusPipeline.from_pretrained(
                self.edit_model_path,
                transformer=transformer,
                scheduler=scheduler,
                torch_dtype=self.dtype
            )

    def _get_scheduler_config(self):
        """Return scheduler configuration for image editing."""
        return  {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),  # We use shift=3 in distillation
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),  # We use shift=3 in distillation
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,  # set shift_terminal to None
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }

    def _prepare_input_image(self, image: Image, pixels: int = INPUT_IMAGE_SIZE):
        total = int(pixels)

        scale_by = math.sqrt(total / (image.width * image.height))
        width = round(image.width * scale_by)
        height = round(image.height * scale_by)

        return image.resize((width, height), Image.Resampling.LANCZOS)

    def _text_prompting_to_embedded(self, text_prompting: TextPrompting, images: Iterable[Image.Image]) -> EmbeddedPrompting:
        # 1. Create a list of prompt_embeds and masks
        embeds_list = []
        masks_list = []

        images = list(self._prepare_input_image(img, self.CONDITION_IMAGE_SIZE) for img in images)
        
        for p in text_prompting.prompt:
            # encode_prompt returns tensors of shape [batch_size_per_prompt, seq_len, hidden_dim]
            e, m = self.pipe.encode_prompt(
                prompt=p,
                image=images,
            )
            embeds_list.append(e)
            masks_list.append(m)

        # Total batch = sum of all individual batches
        total_batch = sum(e.shape[0] for e in embeds_list)
        # Max sequence length across this specific set of prompts
        max_seq_len = max(e.shape[1] for e in embeds_list)
        hidden_dim = embeds_list[0].shape[2]

        # Shape: [total_batch, max_seq_len, hidden_dim]
        output_embeds = embeds_list[0].new_zeros(total_batch, max_seq_len, hidden_dim)
        # Shape: [total_batch, max_seq_len]
        output_masks = masks_list[0].new_zeros(total_batch, max_seq_len) 

        # In for loop copy contents
        current_idx = 0
        for e, m in zip(embeds_list, masks_list):
            b, s, _ = e.shape
            output_embeds[current_idx : current_idx + b, :s, :] = e
            output_masks[current_idx : current_idx + b, :s] = m
            current_idx += b

        # create EmbeddedPrompting
        embedded_prompting = EmbeddedPrompting(prompt_embeds=output_embeds, prompt_embeds_mask = output_masks)
        return embedded_prompting



    def _run_model_pipe(self, seed: Optional[int] = None, **kwargs):
        if seed:
            kwargs.update(dict(generator=torch.Generator(device=self.device).manual_seed(seed)))
        image = kwargs.pop("image", self._empty_image)
        result = self.pipe(
                image=image,
                **self.pipe_config,
                **kwargs)
        return result
    
    def _run_edit_pipe(self,
                       prompt_images: Iterable[Image.Image],
                       seed: Optional[int] = None,
                       **kwargs):
        prompt_images = list(self._prepare_input_image(prompt_image) for prompt_image in prompt_images)

        return self._run_model_pipe(seed=seed, image=prompt_images, **kwargs)
    
    
    def edit_image(self, prompt_image: Image.Image | Iterable[Image.Image], seed: int, prompting: Prompting, encode_prompt: bool = True):
        """ 
        Edit the image using Qwen Edit.

        Args:
            prompt_image: The prompt image to edit.
            reference_image: The reference image to edit.

        Returns:
            The edited image.
        """
        if self.pipe is None:
            logger.error("Edit Model is not loaded")
            raise RuntimeError("Edit Model is not loaded")
        
        try:
            start_time = time.time()

            prompt_images = list(prompt_image) if isinstance(prompt_image, Iterable) else [prompt_image]
            if encode_prompt and isinstance(prompting, TextPrompting):
                prompting = self._text_prompting_to_embedded(prompting, prompt_images)

            prompting_args = prompting.model_dump()
            
            # Run the edit pipe
            result = self._run_edit_pipe(prompt_images=prompt_images,
                                        **prompting_args,
                                        seed=seed)
            
            generation_time = time.time() - start_time
            
            results = tuple(result.images)
            
            logger.success(f"Edited image generated in {generation_time:.2f}s, Size: {results[0].size}, Seed: {seed}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            raise e
