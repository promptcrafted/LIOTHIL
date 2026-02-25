"""Training orchestrator — the main state machine.

Sequences through training phases: unified → fork → expert_1 → expert_2.
Within each phase, the inner loop is identical: sample batch → noise →
forward → loss → backward → step.

The orchestrator manages:
    - Phase sequencing from resolved TrainingPhase list
    - LoRA creation, fork, and management via PEFT bridge
    - Per-phase optimizer and scheduler lifecycle
    - Checkpoint saving and resumption
    - Sampling orchestration
    - Logging and metrics

What changes between phases:
    - Which LoRA is being trained
    - Which optimizer/scheduler are active
    - Which timesteps contribute to loss (expert masking)
    - The resolved hyperparameters

Model-specific operations (load model, forward pass, inference) are
delegated to ModelBackend and InferencePipeline protocols.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from dimljus.training.checkpoint import CheckpointManager, TrainingState
from dimljus.training.errors import (
    DimljusTrainingError,
    ModelBackendError,
)
from dimljus.training.logger import TrainingLogger
from dimljus.training.lora import LoRAState, merge_experts
from dimljus.training.metrics import MetricsTracker
from dimljus.training.optimizer import build_optimizer, build_scheduler, compute_total_steps
from dimljus.training.phase import PhaseType, TrainingPhase, resolve_phases
from dimljus.training.sampler import SamplingEngine


# Default gradient norm limit — prevents training instability from spikes
_DEFAULT_MAX_GRAD_NORM = 1.0


class TrainingOrchestrator:
    """The main training state machine.

    Orchestrates the full training pipeline:
    1. Load config → resolve phases
    2. Print training plan
    3. Load model via ModelBackend
    4. Create LoRA via PEFT bridge
    5. Load cached dataset via DataLoader
    6. Iterate through phases (unified → fork → experts)
    7. Save final merged LoRA

    Two execution modes:
    - Real training (dataset provided): full GPU pipeline with PEFT LoRA,
      mixed precision, gradient accumulation, optimizer/scheduler.
    - Mock/dry run (dataset=None): phase counting only, no GPU operations.
      Used by Phase 7 tests and the CLI 'plan' command.

    Args:
        config: DimljusTrainingConfig instance.
        model_backend: ModelBackend protocol implementation.
        inference_pipeline: InferencePipeline for sampling (optional).
    """

    def __init__(
        self,
        config: Any,
        model_backend: Any,
        inference_pipeline: Any = None,
    ) -> None:
        self._config = config
        self._backend = model_backend
        self._pipeline = inference_pipeline

        # Resolve phases upfront — catch config errors before training
        self._phases = resolve_phases(config)

        # Initialize subsystems
        self._checkpoint_mgr = CheckpointManager(
            output_dir=config.save.output_dir,
            name=config.save.name,
            max_checkpoints=config.save.max_checkpoints,
        )
        self._logger = TrainingLogger(
            backends=config.logging.backends,
            output_dir=config.save.output_dir,
            wandb_project=config.logging.wandb_project,
            wandb_entity=config.logging.wandb_entity,
            wandb_run_name=config.logging.wandb_run_name,
            log_every_n_steps=config.logging.log_every_n_steps,
        )
        self._metrics = MetricsTracker()
        self._sampler = SamplingEngine(
            enabled=config.sampling.enabled,
            every_n_epochs=config.sampling.every_n_epochs,
            prompts=config.sampling.prompts,
            negative_prompt=config.sampling.neg,
            seed=config.sampling.seed,
            walk_seed=config.sampling.walk_seed,
            num_inference_steps=config.sampling.sample_steps,
            guidance_scale=config.sampling.guidance_scale,
            sample_dir=config.sampling.sample_dir or str(
                Path(config.save.output_dir) / "samples"
            ),
            skip_phases=getattr(config.sampling, "skip_phases", None),
        )

        # State
        self._global_step = 0
        self._model: Any = None
        self._unified_lora: LoRAState | None = None
        self._high_noise_lora: LoRAState | None = None
        self._low_noise_lora: LoRAState | None = None

    @property
    def phases(self) -> list[TrainingPhase]:
        """The resolved training phases."""
        return self._phases

    @property
    def global_step(self) -> int:
        """Global training step counter."""
        return self._global_step

    def run(self, dataset: Any = None, dry_run: bool = False) -> None:
        """Execute the full training pipeline.

        Args:
            dataset: CachedLatentDataset instance. Required unless dry_run.
            dry_run: If True, resolve phases and print plan without training.
        """
        # Print the training plan
        self._logger.print_training_plan(self._phases)

        if dry_run:
            return

        # Ensure output directories exist
        self._checkpoint_mgr.ensure_dirs()

        # Check for resumption
        resume_point = self._checkpoint_mgr.find_resume_point(self._phases)
        start_phase_idx = 0
        start_epoch = 0

        if resume_point is not None:
            start_phase_idx, start_epoch, state = resume_point
            self._global_step = state.global_step

            # Restore LoRA states from saved paths
            if state.unified_lora_path:
                self._unified_lora = LoRAState.load(state.unified_lora_path)
            if state.high_noise_lora_path:
                self._high_noise_lora = LoRAState.load(state.high_noise_lora_path)
            if state.low_noise_lora_path:
                self._low_noise_lora = LoRAState.load(state.low_noise_lora_path)

        # Load model
        self._model = self._backend.load_model(self._config.model)
        if self._config.training.gradient_checkpointing:
            self._backend.setup_gradient_checkpointing(self._model)

        # Execute phases
        for phase_idx in range(start_phase_idx, len(self._phases)):
            phase = self._phases[phase_idx]
            epoch_start = start_epoch if phase_idx == start_phase_idx else 0

            self._execute_phase(
                phase=phase,
                phase_index=phase_idx,
                dataset=dataset,
                start_epoch=epoch_start,
            )

            # Fork after unified phase if next phase is an expert phase
            if (
                phase.phase_type == PhaseType.UNIFIED
                and phase_idx + 1 < len(self._phases)
                and self._phases[phase_idx + 1].active_expert is not None
                and self._unified_lora is not None
            ):
                self._logger.log_fork()
                high, low = self._unified_lora.fork()
                self._high_noise_lora = high
                self._low_noise_lora = low

        # Save final merged LoRA
        self._save_final()

        # Cleanup inference pipeline if loaded (free VRAM)
        if self._pipeline is not None and hasattr(self._pipeline, "cleanup"):
            self._pipeline.cleanup()

        # Close logger
        self._logger.close()

    # ------------------------------------------------------------------
    # Phase execution
    # ------------------------------------------------------------------

    def _execute_phase(
        self,
        phase: TrainingPhase,
        phase_index: int,
        dataset: Any,
        start_epoch: int = 0,
    ) -> None:
        """Execute one training phase.

        For real training (dataset provided): creates LoRA via PEFT bridge,
        builds optimizer/scheduler, runs epochs with DataLoader, extracts
        LoRA weights after completion.

        For mock/dry run (dataset=None): runs through epoch counting with
        zero loss, no GPU operations.

        Args:
            phase: The resolved TrainingPhase.
            phase_index: Index into self._phases.
            dataset: CachedLatentDataset (None for dry run).
            start_epoch: Epoch to start from (for resumption).
        """
        self._logger.log_phase_start(phase, phase_index)
        self._metrics.start_phase(phase.phase_type)

        # Get the active LoRA state for this phase (may be None for first phase)
        active_lora = self._get_active_lora(phase)

        # Get noise schedule from model backend
        noise_schedule = self._backend.get_noise_schedule()

        # GPU training setup (only when real dataset provided)
        optimizer = None
        scheduler = None
        if dataset is not None:
            # Switch expert model if needed (MoE phases)
            self._ensure_expert_model(phase)

            # Create LoRA adapter on model, inject weights if available
            active_lora = self._setup_phase_lora(phase, active_lora)

            # Build optimizer and scheduler with real model parameters
            optimizer, scheduler = self._build_phase_optimizer(phase, dataset)

        # Training loop
        for epoch in range(start_epoch + 1, phase.max_epochs + 1):
            self._metrics.set_epoch(epoch)

            # Run one epoch
            epoch_loss = self._run_epoch(
                phase=phase,
                dataset=dataset,
                active_lora=active_lora,
                noise_schedule=noise_schedule,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            # Save checkpoint at interval
            if (
                epoch % self._config.save.save_every_n_epochs == 0
                or epoch == phase.max_epochs
            ):
                self._save_checkpoint(phase, epoch, active_lora)

            # Generate samples
            if self._sampler.should_sample(epoch, phase.phase_type):
                self._generate_samples(phase, epoch, active_lora)

            # Save training state for resumption
            self._save_training_state(phase, phase_index, epoch)

        # GPU teardown: extract LoRA weights, remove PEFT wrapper
        if dataset is not None:
            active_lora = self._teardown_phase_lora(phase, active_lora)

        self._logger.log_phase_end(phase, phase_index)

        # Update stored LoRA reference
        self._update_lora_state(phase, active_lora)

    # ------------------------------------------------------------------
    # LoRA lifecycle (GPU — PEFT bridge)
    # ------------------------------------------------------------------

    def _ensure_expert_model(self, phase: TrainingPhase) -> None:
        """Ensure the correct expert model is loaded for MoE phases.

        For MoE models, each expert phase needs a specific transformer
        (high_noise vs low_noise subfolder). If the wrong expert is
        currently loaded, remove any LoRA, then load the correct one.

        For non-MoE models or unified phases, this is a no-op.
        """
        if phase.active_expert is None:
            return

        # Check if backend tracks current expert
        current_expert = getattr(self._backend, "current_expert", None)
        if current_expert == phase.active_expert:
            return

        # Remove any existing LoRA wrapper before switching
        try:
            from dimljus.training.wan.modules import remove_lora_from_model
            self._model = remove_lora_from_model(self._model)
        except ImportError:
            pass

        # Load the correct expert model
        self._model = self._backend.load_model(
            self._config.model,
            expert=phase.active_expert,
        )
        if self._config.training.gradient_checkpointing:
            self._backend.setup_gradient_checkpointing(self._model)

    def _setup_phase_lora(
        self,
        phase: TrainingPhase,
        active_lora: LoRAState | None,
    ) -> LoRAState:
        """Create LoRA adapter on model for this phase.

        Uses the PEFT bridge from dimljus.training.wan.modules:
        1. Resolve target modules (variant defaults + fork target filtering)
        2. Create LoRA via get_peft_model (wraps model)
        3. Inject existing weights if resuming or post-fork

        Args:
            phase: Current training phase.
            active_lora: Existing LoRA state (None for first phase).

        Returns:
            LoRAState for this phase (existing or newly created).
        """
        from dimljus.training.wan.modules import (
            create_lora_on_model,
            inject_lora_state_dict,
            resolve_target_modules,
        )

        # Resolve final LoRA target modules
        variant_targets = self._backend.get_lora_target_modules()
        target_modules = resolve_target_modules(
            variant_targets=variant_targets,
            fork_targets=phase.fork_targets,
        )

        # Create LoRA adapter (returns PEFT-wrapped model)
        self._model = create_lora_on_model(
            model=self._model,
            target_modules=target_modules,
            rank=self._config.lora.rank,
            alpha=self._config.lora.alpha,
            dropout=phase.lora_dropout,
        )

        # Inject existing weights (resumption or post-fork)
        if active_lora is not None and active_lora.state_dict:
            inject_lora_state_dict(self._model, active_lora.state_dict)
        else:
            # Create new empty LoRAState for this phase
            active_lora = LoRAState(
                state_dict={},
                rank=self._config.lora.rank,
                alpha=self._config.lora.alpha,
                phase_type=phase.phase_type,
            )

        return active_lora

    def _teardown_phase_lora(
        self,
        phase: TrainingPhase,
        active_lora: LoRAState,
    ) -> LoRAState:
        """Extract LoRA weights from model after phase completion.

        Updates the LoRAState with trained weights and removes the PEFT
        wrapper to restore the base model.

        Args:
            phase: Completed training phase.
            active_lora: LoRAState to update with extracted weights.

        Returns:
            Updated LoRAState with trained weights.
        """
        from dimljus.training.wan.modules import (
            extract_lora_state_dict,
            remove_lora_from_model,
        )

        # Extract trained LoRA weights from model
        state_dict = extract_lora_state_dict(self._model)
        active_lora = LoRAState(
            state_dict=state_dict,
            rank=active_lora.rank,
            alpha=active_lora.alpha,
            phase_type=phase.phase_type,
            metadata=active_lora.metadata,
        )

        # Remove PEFT wrapper (restores base model)
        self._model = remove_lora_from_model(self._model)

        return active_lora

    # ------------------------------------------------------------------
    # Optimizer / scheduler construction
    # ------------------------------------------------------------------

    def _build_phase_optimizer(
        self,
        phase: TrainingPhase,
        dataset: Any,
    ) -> tuple[Any, Any]:
        """Build optimizer and scheduler for a training phase.

        Groups parameters by LoRA A-matrix vs B-matrix for LoRA+ support
        (different learning rates). Uses the resolved phase hyperparameters.

        Args:
            phase: Resolved TrainingPhase.
            dataset: CachedLatentDataset (for total step computation).

        Returns:
            Tuple of (optimizer, scheduler).
        """
        # Group parameters: LoRA+ gives B-matrix higher LR
        loraplus_ratio = getattr(self._config.lora, "loraplus_lr_ratio", 1.0)
        a_params = []
        b_params = []
        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue
            if ".lora_B." in name or ".lora_up." in name:
                b_params.append(param)
            else:
                a_params.append(param)

        param_groups: list[dict[str, Any]] = []
        if a_params:
            param_groups.append({"params": a_params, "lr": phase.learning_rate})
        if b_params:
            param_groups.append({
                "params": b_params,
                "lr": phase.learning_rate * loraplus_ratio,
            })

        # Build optimizer
        optimizer_cfg = self._config.optimizer
        optimizer = build_optimizer(
            params=param_groups,
            optimizer_type=phase.optimizer_type,
            learning_rate=phase.learning_rate,
            weight_decay=phase.weight_decay,
            betas=getattr(optimizer_cfg, "betas", None),
            eps=getattr(optimizer_cfg, "eps", 1e-8),
            optimizer_args=getattr(optimizer_cfg, "optimizer_args", None),
        )

        # Compute total optimizer steps for this phase
        num_samples = len(dataset) if hasattr(dataset, "__len__") else 0
        total_steps = compute_total_steps(
            num_samples=num_samples,
            batch_size=phase.batch_size,
            gradient_accumulation_steps=phase.gradient_accumulation_steps,
            max_epochs=phase.max_epochs,
        )

        # Build scheduler
        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_type=phase.scheduler_type,
            total_steps=total_steps,
            warmup_steps=phase.warmup_steps,
            min_lr_ratio=phase.min_lr_ratio,
        )

        return optimizer, scheduler

    # ------------------------------------------------------------------
    # Training loop (inner loop)
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        phase: TrainingPhase,
        dataset: Any,
        active_lora: LoRAState | None,
        noise_schedule: Any,
        optimizer: Any = None,
        scheduler: Any = None,
    ) -> float:
        """Run one training epoch.

        When dataset is provided: uses DataLoader with BucketBatchSampler,
        gradient accumulation, mixed precision, and real gradient updates.

        When dataset is None: returns 0.0 (for dry run / mock testing).

        Args:
            phase: Current training phase.
            dataset: CachedLatentDataset (None for dry run).
            active_lora: Current LoRA state.
            noise_schedule: NoiseSchedule from model backend.
            optimizer: Phase optimizer (None for dry run).
            scheduler: Phase LR scheduler (None for dry run).

        Returns:
            Average loss for the epoch.
        """
        if dataset is None:
            return 0.0

        import torch
        from torch.utils.data import DataLoader

        from dimljus.encoding.dataset import BucketBatchSampler, collate_cached_batch

        # Resolve compute dtype from config
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        mixed_precision = getattr(self._config.training, "mixed_precision", "bf16")
        compute_dtype = dtype_map.get(mixed_precision, torch.bfloat16)

        # Determine device from model parameters
        device = next(
            (p.device for p in self._model.parameters()),
            torch.device("cpu"),
        )

        # Create DataLoader with bucketed batching for uniform dimensions
        batch_sampler = BucketBatchSampler(
            dataset=dataset,
            batch_size=phase.batch_size,
            shuffle=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_cached_batch,
        )

        # Training settings
        grad_accum = phase.gradient_accumulation_steps
        max_grad_norm = getattr(
            self._config.optimizer, "max_grad_norm", _DEFAULT_MAX_GRAD_NORM,
        )

        total_loss = 0.0
        num_steps = 0
        accum_count = 0

        self._model.train()
        optimizer.zero_grad()

        for batch in dataloader:
            # Caption dropout: zero out text embeddings randomly
            batch = self._apply_caption_dropout(batch, phase.caption_dropout_rate)

            # Forward + loss + backward
            loss = self._training_step(
                phase=phase,
                batch=batch,
                noise_schedule=noise_schedule,
                compute_dtype=compute_dtype,
                device=device,
                grad_accum_steps=grad_accum,
            )

            total_loss += loss
            num_steps += 1
            accum_count += 1
            self._global_step += 1

            # Optimizer step after gradient accumulation
            if accum_count >= grad_accum:
                if max_grad_norm is not None and max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self._model.parameters() if p.requires_grad],
                        max_grad_norm,
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                accum_count = 0

            # Update metrics
            current_lr = phase.learning_rate
            if scheduler is not None:
                try:
                    current_lr = scheduler.get_last_lr()[0]
                except Exception:
                    pass
            self._metrics.update(loss=loss, learning_rate=current_lr)

            # Log at interval
            current = self._metrics.get_current()
            if current is not None:
                self._logger.log_step(
                    metrics=current.to_dict(),
                    global_step=self._global_step,
                    phase_type=phase.phase_type,
                )

        # Flush remaining accumulated gradients
        if accum_count > 0:
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self._model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        return total_loss / max(num_steps, 1)

    def _training_step(
        self,
        phase: TrainingPhase,
        batch: dict[str, Any],
        noise_schedule: Any,
        compute_dtype: Any = None,
        device: Any = None,
        grad_accum_steps: int = 1,
    ) -> float:
        """Execute one training step: noise → forward → loss → backward.

        Flow matching training step:
        1. Sample noise and timesteps
        2. Compute noisy latents: (1-t)*clean + t*noise
        3. Compute velocity target: noise - clean
        4. Forward pass through model (with mixed precision)
        5. MSE loss (masked by expert for MoE phases)
        6. Backward pass (scaled for gradient accumulation)

        Args:
            phase: Current training phase.
            batch: Collated batch from DataLoader.
            noise_schedule: NoiseSchedule from model backend.
            compute_dtype: Torch dtype for mixed precision (e.g. torch.bfloat16).
            device: Target device (e.g. torch.device("cuda")).
            grad_accum_steps: Gradient accumulation divisor.

        Returns:
            Unscaled loss value for this step (float).
        """
        import torch
        import torch.nn.functional as F

        latents = batch.get("latent")
        if latents is None:
            return 0.0

        # Move latents to device and compute dtype
        if hasattr(latents, "to"):
            latents = latents.to(device=device, dtype=compute_dtype)

        batch_size = latents.shape[0]

        # 1. Sample pure noise (same shape and dtype as latents)
        noise = torch.randn_like(latents)

        # 2. Sample timesteps (numpy array in [0, 1])
        timesteps_np = noise_schedule.sample_timesteps(
            batch_size=batch_size,
            strategy=self._config.training.timestep_sampling,
            flow_shift=self._config.model.flow_shift or 3.0,
        )
        timesteps = torch.from_numpy(timesteps_np).to(
            device=device, dtype=compute_dtype,
        )

        # 3. Compute noisy latents: (1-t)*clean + t*noise
        t_broadcast = timesteps.reshape(-1, 1, 1, 1, 1)
        noisy_latents = (1.0 - t_broadcast) * latents + t_broadcast * noise

        # 4. Compute velocity target: noise - clean
        target = noise - latents

        # 5. Prepare model-specific inputs (handles text emb, ref image, etc.)
        model_inputs = self._backend.prepare_model_inputs(
            batch=batch,
            timesteps=timesteps,
            noisy_latents=noisy_latents,
        )

        # 6. Forward pass with mixed precision
        device_type = str(device).split(":")[0]
        if compute_dtype is not None and device_type != "cpu":
            with torch.amp.autocast(device_type=device_type, dtype=compute_dtype):
                prediction = self._backend.forward(self._model, **model_inputs)
        else:
            prediction = self._backend.forward(self._model, **model_inputs)

        # 7. Compute MSE loss in float32 for numerical stability
        prediction = prediction.float()
        target = target.float()
        loss = F.mse_loss(prediction, target, reduction="none")
        # Mean over spatial dims → per-sample loss [B]
        loss = loss.mean(dim=list(range(1, loss.ndim)))

        # 8. Apply expert mask (only for expert phases with boundary)
        if phase.boundary_ratio is not None and phase.active_expert is not None:
            high_mask, low_mask = self._backend.get_expert_mask(
                timesteps_np, phase.boundary_ratio,
            )
            if phase.phase_type == PhaseType.HIGH_NOISE:
                mask = torch.from_numpy(high_mask).float().to(device)
            else:
                mask = torch.from_numpy(low_mask).float().to(device)
            loss = loss * mask

        # 9. Mean across batch
        loss = loss.mean()

        # 10. Scale for gradient accumulation and backward
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        return loss.item()

    # ------------------------------------------------------------------
    # Caption dropout
    # ------------------------------------------------------------------

    def _apply_caption_dropout(
        self,
        batch: dict[str, Any],
        dropout_rate: float,
    ) -> dict[str, Any]:
        """Apply caption dropout to a batch.

        Zeroes out text embeddings with probability dropout_rate, forcing
        the model to rely on visual control signals instead of text.

        Args:
            batch: Collated batch dict.
            dropout_rate: Probability of dropping each sample's caption.

        Returns:
            Batch with dropped captions (modified in-place).
        """
        if dropout_rate <= 0.0:
            return batch

        text_emb = batch.get("text_emb")
        text_mask = batch.get("text_mask")

        if text_emb is None or not hasattr(text_emb, "shape"):
            return batch

        # Per-sample dropout
        batch_size = text_emb.shape[0] if text_emb.ndim >= 2 else 1
        for i in range(batch_size):
            if random.random() < dropout_rate:
                text_emb[i].zero_()
                if text_mask is not None and hasattr(text_mask, "__setitem__"):
                    text_mask[i].zero_()

        batch["text_emb"] = text_emb
        if text_mask is not None:
            batch["text_mask"] = text_mask
        return batch

    # ------------------------------------------------------------------
    # LoRA state helpers
    # ------------------------------------------------------------------

    def _get_active_lora(self, phase: TrainingPhase) -> LoRAState | None:
        """Get the LoRA state that should be trained in this phase.

        Args:
            phase: Current training phase.

        Returns:
            The active LoRA state, or None if not yet created.
        """
        if phase.phase_type == PhaseType.UNIFIED:
            return self._unified_lora
        elif phase.phase_type == PhaseType.HIGH_NOISE:
            return self._high_noise_lora
        elif phase.phase_type == PhaseType.LOW_NOISE:
            return self._low_noise_lora
        return None

    def _update_lora_state(self, phase: TrainingPhase, lora: LoRAState | None) -> None:
        """Update the stored LoRA reference after a phase completes.

        Args:
            phase: Completed training phase.
            lora: The LoRA state that was trained.
        """
        if phase.phase_type == PhaseType.UNIFIED:
            self._unified_lora = lora
        elif phase.phase_type == PhaseType.HIGH_NOISE:
            self._high_noise_lora = lora
        elif phase.phase_type == PhaseType.LOW_NOISE:
            self._low_noise_lora = lora

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        phase: TrainingPhase,
        epoch: int,
        lora: LoRAState | None,
    ) -> None:
        """Save a checkpoint at the current epoch.

        Args:
            phase: Current training phase.
            epoch: Current epoch number.
            lora: LoRA state to save.
        """
        if lora is None:
            return

        path = self._checkpoint_mgr.checkpoint_path(phase.phase_type, epoch)
        lora.save(
            path,
            extra_metadata={
                "epoch": str(epoch),
                "global_step": str(self._global_step),
            },
        )
        self._logger.log_checkpoint_saved(path, phase.phase_type, epoch)

        # Prune old checkpoints
        self._checkpoint_mgr.prune_checkpoints(phase.phase_type)

    def _save_training_state(
        self,
        phase: TrainingPhase,
        phase_index: int,
        epoch: int,
    ) -> None:
        """Save training state for resumption.

        Args:
            phase: Current training phase.
            phase_index: Index into the phases list.
            epoch: Current epoch number.
        """
        state = TrainingState(
            phase_index=phase_index,
            phase_type=phase.phase_type.value,
            epoch=epoch,
            global_step=self._global_step,
        )

        # Record latest checkpoint paths
        for pt in PhaseType:
            latest = self._checkpoint_mgr.find_latest_checkpoint(pt)
            if latest is not None:
                path_str = str(latest)
                if pt == PhaseType.UNIFIED:
                    state.unified_lora_path = path_str
                elif pt == PhaseType.HIGH_NOISE:
                    state.high_noise_lora_path = path_str
                elif pt == PhaseType.LOW_NOISE:
                    state.low_noise_lora_path = path_str

        self._checkpoint_mgr.save_training_state(state)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _generate_samples(
        self,
        phase: TrainingPhase,
        epoch: int,
        lora: LoRAState | None,
    ) -> None:
        """Generate sample previews.

        Args:
            phase: Current training phase.
            epoch: Current epoch number.
            lora: Current LoRA state for this phase.
        """
        if self._pipeline is None or lora is None:
            return

        # For expert phases, resolve partner LoRA
        partner_path = self._sampler.resolve_partner_lora(
            active_expert=phase.active_expert,
            high_noise_path=self._checkpoint_mgr.find_latest_checkpoint(PhaseType.HIGH_NOISE),
            low_noise_path=self._checkpoint_mgr.find_latest_checkpoint(PhaseType.LOW_NOISE),
            unified_path=self._checkpoint_mgr.find_latest_checkpoint(PhaseType.UNIFIED),
        )

        try:
            samples = self._sampler.generate_samples(
                pipeline=self._pipeline,
                model=self._model,
                lora_state_dict=lora.state_dict,
                phase_type=phase.phase_type,
                epoch=epoch,
            )
            for i, path in enumerate(samples):
                self._logger.log_sample_generated(path, i)
        except Exception:
            # Sampling failure shouldn't crash training
            pass

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    def _save_final(self) -> None:
        """Save the final merged LoRA for inference.

        For MoE models: merge high-noise and low-noise experts.
        For single-expert: save the unified LoRA as-is.
        """
        # If both experts exist, merge them
        if self._high_noise_lora is not None and self._low_noise_lora is not None:
            try:
                merged = merge_experts(self._high_noise_lora, self._low_noise_lora)
                merged.save(self._checkpoint_mgr.final_path())
            except Exception as e:
                print(
                    f"  Warning: Expert merge failed ({e}). "
                    f"Individual phase checkpoints are still available in: "
                    f"{self._checkpoint_mgr._output_dir}"
                )

        # If only unified exists, save it as final
        elif self._unified_lora is not None:
            self._unified_lora.save(self._checkpoint_mgr.final_path())
