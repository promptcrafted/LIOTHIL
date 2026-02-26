"""WanInferencePipeline — memory-efficient sample generation during training.

Generates preview videos during training using the already-loaded training
model. Avoids loading a second copy of the 14B transformer by reusing the
training model directly.

Memory strategy:
    1. Load T5 text encoder temporarily → encode all prompts → free T5
    2. Build diffusers WanPipeline from components:
       - transformer = the training model (already on GPU with LoRA)
       - VAE = loaded from disk (small: ~243MB)
       - scheduler = FlowMatchEulerDiscreteScheduler (no GPU memory)
    3. Generate using prompt_embeds (pre-encoded) → skip T5 entirely
    4. Free VAE + pipeline after generation

This keeps peak VRAM usage at training_model + VAE (~0.2GB overhead) rather
than training_model + full_pipeline (~37GB overhead).

Supports both individual safetensors files and Diffusers directories.

Requires: torch, diffusers, transformers (the 'wan' dependency group).
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from dimljus.training.errors import SamplingError


def _clean_gpu_memory() -> None:
    """Force Python GC and clear CUDA cache."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class WanInferencePipeline:
    """InferencePipeline for Wan model sample generation during training.

    Uses the training model directly (no second copy). Loads T5 and VAE
    temporarily for encoding/decoding, freeing them afterward to minimize
    VRAM usage.

    Supports two model loading modes:
        - Individual safetensors: provide vae_path and t5_path
        - Diffusers directory: provide diffusers_path (fallback)

    Args:
        vae_path: Path to VAE safetensors file (individual file mode).
        t5_path: Path to T5 encoder weights (.pth or .safetensors).
        diffusers_path: Path to Diffusers model directory (fallback).
            Used when individual file paths are not provided.
        is_i2v: Whether this is an I2V pipeline.
        dtype: Tensor dtype for inference ('bf16', 'fp16', 'fp32').
        device: Target device ('cuda', 'cpu').
    """

    def __init__(
        self,
        vae_path: str | None = None,
        t5_path: str | None = None,
        diffusers_path: str | None = None,
        is_i2v: bool = False,
        dtype: str = "bf16",
        device: str = "cuda",
    ) -> None:
        self._vae_path = vae_path
        self._t5_path = t5_path
        self._diffusers_path = diffusers_path
        self._is_i2v = is_i2v
        self._dtype_str = dtype
        self._device = device

        # Cached prompt embeddings (populated on first generate call)
        self._cached_prompt_embeds: dict[str, Any] = {}

    def _get_torch_dtype(self) -> Any:
        """Map dtype string to torch dtype."""
        import torch
        return {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(self._dtype_str, torch.bfloat16)

    def _encode_prompt(self, prompt: str) -> Any:
        """Encode a single prompt using T5, with caching.

        Loads the T5 encoder on first call, encodes the prompt, and
        caches the result. The encoder is freed after all prompts in
        a generation batch are encoded.

        Returns:
            Tensor of shape [1, seq_len, hidden_dim].
        """
        if prompt in self._cached_prompt_embeds:
            return self._cached_prompt_embeds[prompt]

        raise SamplingError(
            "Prompt not pre-encoded. Call _precompute_embeddings() first."
        )

    def _precompute_embeddings(
        self, prompts: list[str], negative_prompt: str = ""
    ) -> None:
        """Pre-encode all prompts using T5, then free the encoder.

        This is the memory-efficient approach: load T5 once, encode
        everything, then completely free the ~10GB encoder before
        starting the diffusion loop.

        Args:
            prompts: All positive prompts to encode.
            negative_prompt: The negative prompt (shared across all samples).
        """
        # Skip if all prompts already cached
        all_texts = set(prompts)
        if negative_prompt:
            all_texts.add(negative_prompt)

        uncached = [t for t in all_texts if t not in self._cached_prompt_embeds]
        if not uncached:
            return

        import torch

        dtype = self._get_torch_dtype()

        try:
            # Load T5 encoder and tokenizer
            tokenizer, text_encoder = self._load_t5()
            text_encoder = text_encoder.to(self._device, dtype=dtype)
            text_encoder.eval()

            print("  Encoding sampling prompts...")
            with torch.no_grad():
                for text in uncached:
                    tokens = tokenizer(
                        text,
                        max_length=512,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    input_ids = tokens.input_ids.to(self._device)
                    attention_mask = tokens.attention_mask.to(self._device)

                    output = text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    # Get the last hidden state
                    if hasattr(output, "last_hidden_state"):
                        embeds = output.last_hidden_state
                    else:
                        embeds = output[0]

                    # Store on CPU to save GPU memory
                    self._cached_prompt_embeds[text] = embeds.cpu()

            print(f"  Encoded {len(uncached)} prompts")

        finally:
            # Free T5 regardless of success/failure
            del text_encoder, tokenizer
            _clean_gpu_memory()

    def _load_t5(self) -> tuple[Any, Any]:
        """Load T5 tokenizer and encoder model.

        Supports:
            - Individual .pth file (torch.load)
            - Individual .safetensors file
            - Diffusers directory (subfolder 'text_encoder')

        Returns:
            (tokenizer, text_encoder) tuple.
        """
        from transformers import AutoTokenizer

        if self._t5_path and Path(self._t5_path).exists():
            # Individual file mode — use the same loading as text_encoder.py
            return self._load_t5_from_file(self._t5_path)
        elif self._diffusers_path:
            # Diffusers directory mode
            from transformers import UMT5EncoderModel
            tokenizer = AutoTokenizer.from_pretrained(
                self._diffusers_path, subfolder="tokenizer"
            )
            text_encoder = UMT5EncoderModel.from_pretrained(
                self._diffusers_path, subfolder="text_encoder"
            )
            return tokenizer, text_encoder
        else:
            raise SamplingError(
                "No T5 encoder path available for sampling. "
                "Set model.t5 (individual file) or model.path (Diffusers directory) "
                "in your training config."
            )

    def _load_t5_from_file(self, t5_path: str) -> tuple[Any, Any]:
        """Load T5 from an individual weight file (.pth or .safetensors).

        Uses the same approach as dimljus.encoding.text_encoder:
        UMT5Config → empty model → load_state_dict.

        Returns:
            (tokenizer, text_encoder) tuple.
        """
        import torch
        from transformers import AutoTokenizer, UMT5EncoderModel, UMT5Config

        # Load tokenizer from HuggingFace (small download, cached)
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

        # Build empty model from config, then load weights
        config = UMT5Config.from_pretrained("google/umt5-xxl")
        text_encoder = UMT5EncoderModel(config)

        path = Path(t5_path)
        if path.suffix == ".pth":
            state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        else:
            from safetensors.torch import load_file
            state_dict = load_file(str(path))

        text_encoder.load_state_dict(state_dict, strict=False)
        return tokenizer, text_encoder

    def _load_vae(self) -> Any:
        """Load the Wan VAE for decoding latents to video frames.

        Returns:
            AutoencoderKLWan instance.
        """
        import torch
        from diffusers.models import AutoencoderKLWan

        dtype = self._get_torch_dtype()

        if self._vae_path and Path(self._vae_path).exists():
            vae = AutoencoderKLWan.from_single_file(
                self._vae_path, torch_dtype=dtype
            )
        elif self._diffusers_path:
            vae = AutoencoderKLWan.from_pretrained(
                self._diffusers_path, subfolder="vae", torch_dtype=dtype
            )
        else:
            raise SamplingError(
                "No VAE path available for sampling. "
                "Set model.vae (individual file) or model.path (Diffusers directory) "
                "in your training config."
            )

        return vae.to(self._device)

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
        """Generate a sample video using the training model.

        Memory-efficient approach:
        1. Pre-encode prompts with T5 (loaded temporarily, then freed)
        2. Build a minimal pipeline: training model + VAE + scheduler
        3. Generate using prompt_embeds (no T5 needed during denoising)
        4. Export frames to video and clean up

        Args:
            model: The training transformer (already on GPU with LoRA).
            lora_state_dict: LoRA state dict (unused — LoRA is already
                applied to the model during training).
            prompt: Positive text prompt.
            negative_prompt: Negative text prompt for CFG.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.
            reference_image: Reference image for I2V (PIL Image or path).

        Returns:
            Generated video frames (list of lists of PIL Images).

        Raises:
            SamplingError: If generation fails.
        """
        try:
            import torch
        except ImportError as e:
            raise SamplingError(f"torch required for inference: {e}")

        try:
            # Step 1: Pre-encode prompts (loads T5 temporarily, then frees it)
            all_prompts = [prompt]
            if negative_prompt:
                all_prompts.append(negative_prompt)
            self._precompute_embeddings(all_prompts, negative_prompt)

            # Move cached embeddings to GPU
            prompt_embeds = self._cached_prompt_embeds[prompt].to(
                self._device, dtype=self._get_torch_dtype()
            )
            neg_embeds = None
            if negative_prompt and negative_prompt in self._cached_prompt_embeds:
                neg_embeds = self._cached_prompt_embeds[negative_prompt].to(
                    self._device, dtype=self._get_torch_dtype()
                )

            # Step 2: Build minimal pipeline with the training model
            pipeline = self._build_pipeline(model)

            # Step 3: Switch model to eval mode for inference
            was_training = model.training
            model.eval()

            generator = torch.Generator(device=self._device).manual_seed(seed)

            # Build generation kwargs
            kwargs: dict[str, Any] = {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": neg_embeds,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }

            # Add reference image for I2V
            if self._is_i2v and reference_image is not None:
                kwargs["image"] = reference_image

            with torch.no_grad():
                output = pipeline(**kwargs)

            # Restore training mode
            if was_training:
                model.train()

            # Extract frames
            frames = output.frames if hasattr(output, "frames") else output

            # Clean up pipeline (keeps training model, frees VAE + scheduler)
            # Don't delete model — it's the training model
            pipeline.transformer = None
            del pipeline
            _clean_gpu_memory()

            return frames

        except SamplingError:
            raise
        except torch.cuda.OutOfMemoryError:
            _clean_gpu_memory()
            raise SamplingError(
                "Out of GPU memory during sampling. "
                "This is expected when training uses most of the VRAM. "
                "Set sampling.enabled: false in your config, or reduce "
                "training batch size to free memory for sampling."
            )
        except Exception as e:
            _clean_gpu_memory()
            raise SamplingError(
                f"Video generation failed: {e}\n"
                f"This may indicate insufficient VRAM or model incompatibility."
            ) from e

    def _build_pipeline(self, model: Any) -> Any:
        """Build a diffusers pipeline using the training model and fresh VAE.

        Constructs the pipeline from individual components rather than
        from_pretrained(), so it works with both individual safetensors
        files and Diffusers directories.

        Args:
            model: Training transformer (already on GPU).

        Returns:
            WanPipeline or WanImageToVideoPipeline instance.
        """
        from diffusers import FlowMatchEulerDiscreteScheduler

        # Load VAE (small: ~243MB)
        vae = self._load_vae()

        # Create scheduler with flow_shift=3.0 for Wan models
        # Without shift, the noise schedule is wrong and produces garbage output.
        # ai-toolkit uses flow_shift=3.0, musubi uses shift=3.0
        scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)

        if self._is_i2v:
            from diffusers import WanImageToVideoPipeline
            pipeline = WanImageToVideoPipeline(
                transformer=model,
                vae=vae,
                text_encoder=None,
                tokenizer=None,
                scheduler=scheduler,
            )
        else:
            from diffusers import WanPipeline
            pipeline = WanPipeline(
                transformer=model,
                vae=vae,
                text_encoder=None,
                tokenizer=None,
                scheduler=scheduler,
            )

        return pipeline

    def cleanup(self) -> None:
        """Release GPU memory and clear cached embeddings."""
        self._cached_prompt_embeds.clear()
        _clean_gpu_memory()
