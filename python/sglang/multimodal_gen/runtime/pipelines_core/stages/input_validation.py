# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Input validation stage for diffusion pipelines.
"""
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs import WanI2V480PConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.runtime.models.vision_utils import load_image, load_video
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators,
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import best_output_size

logger = init_logger(__name__)

# Alias for convenience
V = StageValidators


# TODO: since this might change sampling params after logging, should be do this beforehand?


def isotropic_crop_resize_pil(
    image: Image.Image, target_size: tuple[int, int]
) -> Image.Image:
    target_width, target_height = target_size
    orig_width, orig_height = image.size
    target_ratio = target_height / target_width
    orig_ratio = orig_height / orig_width

    if abs(orig_ratio - target_ratio) < 0.01:
        return image.resize((target_width, target_height), Image.LANCZOS)

    if orig_ratio > target_ratio:
        # Image is taller, crop height
        crop_height = int(target_ratio * orig_width)
        crop_width = orig_width
        y0 = (orig_height - crop_height) // 2
        y1 = y0 + crop_height
        x0, x1 = 0, orig_width
    else:
        # Image is wider, crop width
        crop_width = int(orig_height / target_ratio)
        crop_height = orig_height
        x0 = (orig_width - crop_width) // 2
        x1 = x0 + crop_width
        y0, y1 = 0, orig_height

    cropped_image = image.crop((x0, y0, x1, y1))
    resized_image = cropped_image.resize((target_width, target_height), Image.LANCZOS)

    return resized_image


RESOLUTION_PRESETS = {
    "16:9": {
        "480p": (832, 480),
        "580p": (960, 512),
        "720p": (1280, 720),
    },
    "9:16": {
        "480p": (480, 832),
        "580p": (512, 960),
        "720p": (720, 1280),
    },
    "1:1": {
        "480p": (480, 480),
        "580p": (512, 512),
        "720p": (720, 720),
    },
}


def parse_resolution_and_aspect_ratio(
    original_aspect_ratio: float, resolution: str | None, aspect_ratio: str | None
) -> tuple[int, int] | None:
    if resolution is None or aspect_ratio is None:
        return None

    if aspect_ratio == "auto":
        available_resolutions = set()
        for ar in RESOLUTION_PRESETS.values():
            available_resolutions.update(ar.keys())

        if resolution not in available_resolutions:
            logger.warning(
                f"Resolution {resolution} not found in any preset, using default logic"
            )
            return None

        # Compare original aspect ratio with actual aspect ratios at the given resolution
        # Calculate the actual aspect ratio (height/width) for each preset at this resolution
        aspect_ratio_candidates = {}
        for ar_key, ar_presets in RESOLUTION_PRESETS.items():
            if resolution in ar_presets:
                width, height = ar_presets[resolution]
                actual_aspect_ratio = height / width
                aspect_ratio_candidates[ar_key] = actual_aspect_ratio

        # Find the closest aspect ratio based on the actual values at this resolution
        closest_aspect_ratio = min(
            aspect_ratio_candidates.items(),
            key=lambda x: abs(x[1] - original_aspect_ratio),
        )
        aspect_ratio = closest_aspect_ratio[0]
        logger.info(
            f"Based on the original aspect ratio {original_aspect_ratio} and resolution {resolution}, "
            f"the closest aspect ratio is {aspect_ratio} (actual ratio: {closest_aspect_ratio[1]:.4f})"
        )

    if aspect_ratio not in RESOLUTION_PRESETS:
        logger.warning(f"Invalid aspect_ratio: {aspect_ratio}, using default logic")
        return None

    if resolution not in RESOLUTION_PRESETS[aspect_ratio]:
        logger.warning(
            f"Resolution {resolution} not found for aspect_ratio {aspect_ratio}, using default logic"
        )
        return None

    return RESOLUTION_PRESETS[aspect_ratio][resolution]  # (width, height)


class InputValidationStage(PipelineStage):
    """
    Stage for validating and preparing inputs for diffusion pipelines.

    This stage validates that all required inputs are present and properly formatted
    before proceeding with the diffusion process.

    In this stage, input image and output image may be resized
    """

    def __init__(self, vae_image_processor=None):
        super().__init__()
        self.vae_image_processor = vae_image_processor

    def _generate_seeds(self, batch: Req, server_args: ServerArgs):
        """Generate seeds for the inference"""
        seed = batch.seed
        num_videos_per_prompt = batch.num_outputs_per_prompt

        assert seed is not None
        seeds = [seed + i for i in range(num_videos_per_prompt)]
        batch.seeds = seeds

        # Create generators based on generator_device parameter
        # Note: This will overwrite any existing batch.generator
        generator_device = batch.generator_device

        if generator_device == "cpu":
            device_str = "cpu"
        else:
            device_str = current_platform.device_type

        batch.generator = [
            torch.Generator(device_str).manual_seed(seed) for seed in seeds
        ]

    def preprocess_condition_image(
        self,
        batch: Req,
        server_args: ServerArgs,
        condition_image_width,
        condition_image_height,
    ):
        """
        preprocess condition image
        NOTE: condition image resizing is only allowed in InputValidationStage
        """
        if batch.condition_image is not None and (
            server_args.pipeline_config.task_type == ModelTaskType.I2I
            or server_args.pipeline_config.task_type == ModelTaskType.TI2I
        ):
            # calculate new condition image size
            if not isinstance(batch.condition_image, list):
                batch.condition_image = [batch.condition_image]

            processed_images = []
            final_image = batch.condition_image[-1]
            config = server_args.pipeline_config
            config.preprocess_vae_image(batch, self.vae_image_processor)

            for img in batch.condition_image:
                size = config.calculate_condition_image_size(img, img.width, img.height)
                if size is not None:
                    width, height = size
                    img, _ = config.preprocess_condition_image(
                        img, width, height, self.vae_image_processor
                    )

                processed_images.append(img)

            batch.condition_image = processed_images
            calculated_size = config.prepare_calculated_size(final_image)

            # adjust output image size
            if calculated_size is not None:
                calculated_width, calculated_height = calculated_size
                width = batch.width or calculated_width
                height = batch.height or calculated_height
                multiple_of = (
                    server_args.pipeline_config.vae_config.get_vae_scale_factor() * 2
                )
                width = width // multiple_of * multiple_of
                height = height // multiple_of * multiple_of
                batch.width = width
                batch.height = height

        elif server_args.pipeline_config.task_type == ModelTaskType.TI2V:
            # duplicate with vae_image_processor
            # further processing for ti2v task
            if isinstance(
                batch.condition_image, list
            ):  # not support multi image input yet.
                batch.condition_image = batch.condition_image[0]

            img = batch.condition_image
            ih, iw = img.height, img.width
            patch_size = server_args.pipeline_config.dit_config.arch_config.patch_size
            vae_stride = (
                server_args.pipeline_config.vae_config.arch_config.scale_factor_spatial
            )
            dh, dw = patch_size[1] * vae_stride, patch_size[2] * vae_stride
            max_area = 704 * 1280
            ow, oh = best_output_size(iw, ih, dw, dh, max_area)

            scale = max(ow / iw, oh / ih)
            img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
            logger.info("resized img height: %s, img width: %s", img.height, img.width)

            # center-crop
            x1 = (img.width - ow) // 2
            y1 = (img.height - oh) // 2
            img = img.crop((x1, y1, x1 + ow, y1 + oh))
            assert img.width == ow and img.height == oh

            # to tensor
            img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)
            img = img.unsqueeze(0)
            batch.height = oh
            batch.width = ow
            # TODO: should we store in a new field: pixel values?
            batch.condition_image = img

        elif isinstance(server_args.pipeline_config, WanI2V480PConfig):
            # TODO: could we merge with above?
            # resize image only, Wan2.1 I2V
            if isinstance(batch.condition_image, list):
                batch.condition_image = batch.condition_image[
                    0
                ]  # not support multi image input yet.

            logger.info(
                f"original image shape: {condition_image_width}x{condition_image_height}"
            )

            mod_value = (
                server_args.pipeline_config.vae_config.arch_config.scale_factor_spatial
                * server_args.pipeline_config.dit_config.arch_config.patch_size[1]
            )

            logger.info(
                f"resolution: {batch.resolution}, aspect_ratio: {batch.aspect_ratio}"
            )

            target_size = parse_resolution_and_aspect_ratio(
                condition_image_width / condition_image_height,
                batch.resolution,
                batch.aspect_ratio,
            )

            if target_size is not None:
                target_width, target_height = target_size
                height = target_height // mod_value * mod_value
                width = target_width // mod_value * mod_value
                batch.condition_image = isotropic_crop_resize_pil(
                    batch.condition_image, (width, height)
                )
            else:
                max_area = server_args.pipeline_config.max_area
                aspect_ratio = condition_image_height / condition_image_width
                height = (
                    round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                )
                width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
                batch.condition_image = batch.condition_image.resize((width, height))

            logger.info(f"resized image shape: {width}x{height}")

            batch.height, batch.width = height, width

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Validate and prepare inputs.
        """

        self._generate_seeds(batch, server_args)

        # Ensure prompt is properly formatted
        if batch.prompt is None and batch.prompt_embeds is None:
            raise ValueError("Either `prompt` or `prompt_embeds` must be provided")

        # Ensure negative prompt is properly formatted if using classifier-free guidance
        if (
            batch.do_classifier_free_guidance
            and batch.negative_prompt is None
            and batch.negative_prompt_embeds is None
        ):
            raise ValueError(
                "For classifier-free guidance, either `negative_prompt` or "
                "`negative_prompt_embeds` must be provided"
            )

        # Validate number of inference steps
        if batch.num_inference_steps <= 0:
            raise ValueError(
                f"Number of inference steps must be positive, but got {batch.num_inference_steps}"
            )

        # Validate guidance scale if using classifier-free guidance
        if batch.do_classifier_free_guidance and batch.guidance_scale < 0:
            raise ValueError(
                f"Guidance scale must be positive, but got {batch.guidance_scale}"
            )

        # for i2v, get image from image_path
        # @TODO(Wei) hard-coded for wan2.2 5b ti2v for now. Should put this in image_encoding stage
        if batch.image_path is not None:
            if isinstance(batch.image_path, list):
                batch.condition_image = []
                for path in batch.image_path:
                    if path.endswith(".mp4"):
                        image = load_video(path)[0]
                    else:
                        image = load_image(path)
                    batch.condition_image.append(image)

                # Use the first image for size reference
                condition_image_width = batch.condition_image[0].width
                condition_image_height = batch.condition_image[0].height
                batch.original_condition_image_size = (
                    condition_image_width,
                    condition_image_height,
                )
            else:
                if batch.image_path.endswith(".mp4"):
                    image = load_video(batch.image_path)[0]
                else:
                    image = load_image(batch.image_path)
                batch.condition_image = image
                condition_image_width, condition_image_height = (
                    image.width,
                    image.height,
                )
                batch.original_condition_image_size = image.size

            self.preprocess_condition_image(
                batch, server_args, condition_image_width, condition_image_height
            )

        # if height or width is not specified at this point, set default to 720p
        default_height = 720
        default_width = 1280
        if batch.height is None and batch.width is None:
            batch.height = default_height
            batch.width = default_width
        elif batch.height is None:
            batch.height = batch.width * default_height // default_width
        elif batch.width is None:
            batch.width = batch.height * default_width // default_height

        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify input validation stage inputs."""
        result = VerificationResult()
        result.add_check("seed", batch.seed, [V.not_none, V.non_negative_int])
        result.add_check(
            "num_videos_per_prompt", batch.num_outputs_per_prompt, V.positive_int
        )
        result.add_check(
            "prompt_or_embeds",
            None,
            lambda _: V.string_or_list_strings(batch.prompt)
            or V.list_not_empty(batch.prompt_embeds),
        )

        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check(
            "guidance_scale",
            batch.guidance_scale,
            lambda x: not batch.do_classifier_free_guidance or V.non_negative_float(x),
        )
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify input validation stage outputs."""
        result = VerificationResult()
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("seeds", batch.seeds, V.list_not_empty)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        return result
