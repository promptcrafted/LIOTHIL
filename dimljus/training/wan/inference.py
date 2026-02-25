"""WanInferencePipeline — implements InferencePipeline for sample generation.

Uses diffusers WanPipeline (T2V) or WanImageToVideoPipeline (I2V) for
generating preview videos during training. Separate from WanModelBackend
because inference follows a completely different code path: iterative
denoising with a scheduler, classifier-free guidance, and video decoding.

The pipeline is loaded lazily on first generate() call, since sample
generation is optional and most of the training loop doesn't need it.

Requires: torch, diffusers (the 'wan' dependency group).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dimljus.training.errors import SamplingError


class WanInferencePipeline:
    """InferencePipeline implementation for Wan model sample generation.

    Lazy-loads a full diffusers pipeline for generating preview videos.
    Supports both T2V and I2V variants. LoRA weights are applied
    temporarily during generation and removed afterward.

    Args:
        model_path: Path to model directory or HuggingFace ID.
        is_i2v: Whether this is an I2V pipeline.
        dtype: Tensor dtype for inference ('bf16', 'fp16', 'fp32').
        device: Target device ('cuda', 'cpu').
    """

    def __init__(
        self,
        model_path: str,
        is_i2v: bool = False,
        dtype: str = "bf16",
        device: str = "cuda",
    ) -> None:
        self._model_path = model_path
        self._is_i2v = is_i2v
        self._dtype_str = dtype
        self._device = device
        self._pipeline: Any = None  # Lazy loaded

    def _ensure_pipeline(self) -> None:
        """Load the diffusers pipeline if not already loaded.

        Raises:
            SamplingError: If pipeline loading fails.
        """
        if self._pipeline is not None:
            return

        try:
            import torch

            dtype_map = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "fp32": torch.float32,
            }
            dtype = dtype_map.get(self._dtype_str, torch.bfloat16)

            if self._is_i2v:
                from diffusers import WanImageToVideoPipeline
                self._pipeline = WanImageToVideoPipeline.from_pretrained(
                    self._model_path,
                    torch_dtype=dtype,
                )
            else:
                from diffusers import WanPipeline
                self._pipeline = WanPipeline.from_pretrained(
                    self._model_path,
                    torch_dtype=dtype,
                )

            self._pipeline.to(self._device)

        except ImportError as e:
            raise SamplingError(
                f"Required packages not installed for inference: {e}\n"
                f"Install with: pip install 'dimljus[wan]'"
            )
        except Exception as e:
            raise SamplingError(
                f"Failed to load inference pipeline from "
                f"'{self._model_path}': {e}"
            ) from e

    def generate(
        self,
        model: Any,
        lora_state_dict: dict[str, Any] | None,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        seed: int = 42,
        reference_image: Any = None,
    ) -> Any:
        """Generate a sample video using the Wan pipeline.

        Temporarily applies LoRA weights, generates a video, then
        removes the LoRA. Returns the generated frames.

        Args:
            model: The loaded transformer model (may be used to replace
                the pipeline's transformer for current LoRA state).
            lora_state_dict: LoRA weights to apply during generation.
                None = generate without LoRA (baseline comparison).
            prompt: Positive text prompt.
            negative_prompt: Negative text prompt for CFG.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.
            reference_image: Reference image for I2V (PIL Image or path).

        Returns:
            Generated video frames (list of PIL Images or tensor).

        Raises:
            SamplingError: If generation fails.
        """
        self._ensure_pipeline()

        try:
            import torch
        except ImportError as e:
            raise SamplingError(f"torch required for inference: {e}")

        try:
            generator = torch.Generator(device=self._device).manual_seed(seed)

            # Build pipeline kwargs
            kwargs: dict[str, Any] = {
                "prompt": prompt,
                "negative_prompt": negative_prompt if negative_prompt else None,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }

            # Add reference image for I2V
            if self._is_i2v and reference_image is not None:
                kwargs["image"] = reference_image

            # Run generation
            output = self._pipeline(**kwargs)

            # Return the frames
            if hasattr(output, "frames"):
                return output.frames
            return output

        except SamplingError:
            raise
        except Exception as e:
            raise SamplingError(
                f"Video generation failed: {e}\n"
                f"This may indicate insufficient VRAM or model incompatibility."
            ) from e

    def cleanup(self) -> None:
        """Release GPU memory from the inference pipeline."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
