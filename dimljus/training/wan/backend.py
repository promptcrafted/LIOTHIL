"""WanModelBackend — implements the ModelBackend protocol for Wan models.

This is the real model backend that replaces Phase 7's stubs. It handles:
    - Loading WanTransformer3DModel from disk
    - Determining LoRA targets for the variant
    - Computing expert masks from timesteps
    - Preparing inputs for the Wan forward pass
    - Running the forward pass
    - Setting up gradient checkpointing

For MoE models (Wan 2.2), the backend loads one expert at a time.
Expert switching happens at phase transitions — the old expert is
unloaded and the new one loaded from disk.

Requires: torch, diffusers, peft (the 'wan' dependency group).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dimljus.training.errors import ModelBackendError
from dimljus.training.noise import FlowMatchingSchedule, get_expert_masks
from dimljus.training.wan.constants import WAN_EXPERT_SUBFOLDERS, WAN_SINGLE_SUBFOLDER


class WanModelBackend:
    """ModelBackend implementation for all Wan model variants.

    Supports Wan 2.1 T2V, Wan 2.2 T2V, and Wan 2.2 I2V. Architecture
    differences are handled via constructor parameters (is_moe, is_i2v,
    in_channels, etc.) rather than subclassing.

    Args:
        model_id: Human-readable identifier (e.g. 'wan-2.2-t2v-14b').
        model_path: Path to model directory or HuggingFace ID. Used as
            fallback when individual file paths are not provided.
        is_moe: Whether this is an MoE model (Wan 2.2 = True, 2.1 = False).
        is_i2v: Whether this is an I2V model.
        in_channels: Input channel count (16 for T2V, 36 for I2V).
        num_blocks: Number of transformer blocks (40).
        boundary_ratio: SNR boundary for expert routing (None for non-MoE).
        flow_shift: Flow matching shift parameter.
        lora_targets: List of module suffixes for LoRA placement.
        expert_subfolders: Maps expert names to diffusers subfolder paths.
        dit_path: Path to a single safetensors file for non-MoE models.
        dit_high_path: Path to high-noise expert safetensors (MoE only).
        dit_low_path: Path to low-noise expert safetensors (MoE only).
    """

    def __init__(
        self,
        model_id: str,
        model_path: str | None = None,
        is_moe: bool = True,
        is_i2v: bool = False,
        in_channels: int = 16,
        num_blocks: int = 40,
        boundary_ratio: float | None = 0.875,
        flow_shift: float = 3.0,
        lora_targets: list[str] | None = None,
        expert_subfolders: dict[str, str] | None = None,
        dit_path: str | None = None,
        dit_high_path: str | None = None,
        dit_low_path: str | None = None,
    ) -> None:
        self._model_id = model_id
        self._model_path = model_path
        self._is_moe = is_moe
        self._is_i2v = is_i2v
        self._in_channels = in_channels
        self._num_blocks = num_blocks
        self._boundary_ratio = boundary_ratio
        self._flow_shift = flow_shift
        self._lora_targets = lora_targets or []
        self._expert_subfolders = expert_subfolders or {}
        self._noise_schedule = FlowMatchingSchedule(1000)

        # Individual safetensors file paths (alternative to Diffusers directory)
        self._dit_path = dit_path
        self._dit_high_path = dit_high_path
        self._dit_low_path = dit_low_path

        # Track which expert is currently loaded (for MoE switching)
        self._current_expert: str | None = None

    # -- ModelBackend protocol properties --

    @property
    def model_id(self) -> str:
        """Human-readable identifier for this model backend."""
        return self._model_id

    @property
    def supports_moe(self) -> bool:
        """Whether this model has Mixture of Experts architecture."""
        return self._is_moe

    @property
    def supports_reference_image(self) -> bool:
        """Whether this model accepts a reference image as conditioning."""
        return self._is_i2v

    # -- ModelBackend protocol methods --

    def load_model(
        self,
        config: Any,
        expert: str | None = None,
    ) -> Any:
        """Load the Wan transformer model from disk.

        Two loading modes:
        1. **Single-file** — When dit_path (non-MoE) or dit_high_path/
           dit_low_path (MoE) are set, uses from_single_file().
        2. **Diffusers directory** (fallback) — Uses from_pretrained()
           with model_path and a subfolder.

        For MoE models, the expert parameter selects which expert to load.
        For single-expert models, expert is ignored.

        Args:
            config: ModelConfig from training schema (used for precision).
            expert: Which expert to load ('high_noise' or 'low_noise').
                None = load the default/single transformer.

        Returns:
            Loaded WanTransformer3DModel instance.

        Raises:
            ModelBackendError: If loading fails.
        """
        try:
            import torch
            from diffusers import WanTransformer3DModel
        except ImportError as e:
            raise ModelBackendError(
                f"Required packages not installed: {e}\n"
                f"Install with: pip install 'dimljus[wan]'"
            )

        # Determine dtype from config
        dtype_str = getattr(config, "base_model_precision", "bf16")
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        dtype = dtype_map.get(dtype_str, torch.bfloat16)

        # Check for individual safetensors file path
        single_file = self._resolve_single_file_path(expert)

        if single_file is not None:
            # -- Single-file loading --
            try:
                model = WanTransformer3DModel.from_single_file(
                    single_file,
                    torch_dtype=dtype,
                )
            except Exception as e:
                raise ModelBackendError(
                    f"Failed to load Wan model from '{single_file}': {e}\n"
                    f"Check that the file exists and is a valid safetensors "
                    f"checkpoint."
                ) from e
        else:
            # -- Diffusers directory loading (fallback) --
            if expert is not None and expert in self._expert_subfolders:
                subfolder = self._expert_subfolders[expert]
            elif "default" in self._expert_subfolders:
                subfolder = self._expert_subfolders["default"]
            else:
                subfolder = WAN_SINGLE_SUBFOLDER

            if not self._model_path:
                raise ModelBackendError(
                    "No model path configured. Set individual file paths "
                    "(dit_high, dit_low) or a Diffusers directory (path) "
                    "in the model config."
                )

            try:
                model_path = Path(self._model_path)
                if model_path.is_dir():
                    model = WanTransformer3DModel.from_pretrained(
                        str(model_path),
                        subfolder=subfolder,
                        torch_dtype=dtype,
                    )
                else:
                    model = WanTransformer3DModel.from_pretrained(
                        self._model_path,
                        subfolder=subfolder,
                        torch_dtype=dtype,
                    )
            except Exception as e:
                raise ModelBackendError(
                    f"Failed to load Wan model from '{self._model_path}' "
                    f"(subfolder='{subfolder}'): {e}\n"
                    f"Check that the path is correct and the model files "
                    f"exist."
                ) from e

        self._current_expert = expert
        return model

    def _resolve_single_file_path(self, expert: str | None) -> str | None:
        """Determine which single-file path to use, if any.

        For MoE: maps expert name to dit_high_path / dit_low_path.
        For non-MoE: returns dit_path.
        Returns None to signal fallback to from_pretrained().
        """
        if self._is_moe and expert is not None:
            return {"high_noise": self._dit_high_path,
                    "low_noise": self._dit_low_path}.get(expert)
        return self._dit_path

    def get_lora_target_modules(self) -> list[str]:
        """Return the list of module names for LoRA adapter placement.

        Returns:
            List of module name suffixes (e.g. ['attn1.to_q', 'ffn.net.2']).
        """
        return list(self._lora_targets)

    def get_expert_mask(
        self,
        timesteps: Any,
        boundary_ratio: float,
    ) -> tuple[Any, Any]:
        """Compute expert masks for a batch of timesteps.

        Delegates to the noise module's get_expert_masks() function.

        Args:
            timesteps: Timestep values [B], in [0, 1].
            boundary_ratio: SNR boundary between experts.

        Returns:
            Tuple of (high_noise_mask, low_noise_mask), each [B].
        """
        return get_expert_masks(timesteps, boundary_ratio)

    def prepare_model_inputs(
        self,
        batch: dict[str, Any],
        timesteps: Any,
        noisy_latents: Any,
    ) -> dict[str, Any]:
        """Prepare inputs for the Wan transformer forward pass.

        Converts the generic batch dict into Wan-specific forward kwargs.
        Handles text embedding injection, reference image concatenation
        (for I2V), and timestep formatting.

        Args:
            batch: Collated batch from CachedLatentDataset. Keys:
                - 'latent': clean latent tensor [B, C, F, H, W]
                - 'text_emb': T5 text embedding [B, seq, 4096] or None
                - 'text_mask': T5 attention mask [B, seq] or None
                - 'reference': encoded reference image (I2V only) or None
            timesteps: Sampled timesteps [B], in [0, 1].
            noisy_latents: Noise-corrupted latents [B, C, F, H, W].

        Returns:
            Dict of kwargs for the model forward pass.
        """
        import torch

        inputs: dict[str, Any] = {}

        # Hidden states: the noisy latents (or concatenated with reference for I2V)
        hidden_states = noisy_latents

        if self._is_i2v and batch.get("reference") is not None:
            # For I2V: concatenate reference encoding with noisy latents
            # along the channel dimension. The reference has already been
            # VAE-encoded and includes the mask channels.
            reference = batch["reference"]
            if reference is not None:
                hidden_states = torch.cat([noisy_latents, reference], dim=1)

        inputs["hidden_states"] = hidden_states

        # Timestep: Wan expects integer timesteps scaled to [0, num_timesteps]
        # Our timesteps are float [0, 1]; scale to [0, 1000]
        inputs["timestep"] = timesteps * self._noise_schedule.num_timesteps

        # Text conditioning
        if batch.get("text_emb") is not None:
            inputs["encoder_hidden_states"] = batch["text_emb"]
        if batch.get("text_mask") is not None:
            inputs["encoder_attention_mask"] = batch["text_mask"]

        return inputs

    def forward(self, model: Any, **inputs: Any) -> Any:
        """Run one forward pass through the Wan transformer.

        Args:
            model: Loaded WanTransformer3DModel (possibly PEFT-wrapped).
            **inputs: Model inputs from prepare_model_inputs().

        Returns:
            Model prediction tensor [B, C, F, H, W].

        Raises:
            ModelBackendError: If the forward pass fails.
        """
        try:
            output = model(**inputs)
            # diffusers returns a Transformer2DModelOutput with .sample
            if hasattr(output, "sample"):
                return output.sample
            return output
        except Exception as e:
            raise ModelBackendError(
                f"Forward pass failed: {e}\n"
                f"This may indicate a shape mismatch between the input "
                f"tensors and the model's expected input format."
            ) from e

    def setup_gradient_checkpointing(self, model: Any) -> None:
        """Enable gradient checkpointing on the Wan transformer.

        Trades compute for VRAM by recomputing activations during the
        backward pass instead of storing them. Essential for video
        training where activation memory is enormous.

        Args:
            model: Loaded WanTransformer3DModel.
        """
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()
        elif hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    def get_noise_schedule(self) -> FlowMatchingSchedule:
        """Return the flow matching noise schedule for Wan.

        All Wan models use 1000-step flow matching with the same
        interpolation formula: noisy = (1-t)*clean + t*noise.

        Returns:
            FlowMatchingSchedule instance with 1000 timesteps.
        """
        return self._noise_schedule

    # -- Wan-specific methods (not in protocol) --

    @property
    def current_expert(self) -> str | None:
        """Which expert is currently loaded (None if single-transformer)."""
        return self._current_expert

    @property
    def boundary_ratio(self) -> float | None:
        """Default SNR boundary for expert routing."""
        return self._boundary_ratio

    @property
    def flow_shift(self) -> float:
        """Flow matching shift parameter."""
        return self._flow_shift
