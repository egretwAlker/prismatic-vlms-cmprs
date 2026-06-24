"""
ddp.py

Core class definition for a strategy implementing Torch native Distributed Data Parallel Training; note that on most
GPU hardware and LLM backbones >= 5-7B parameters, DDP training will OOM, which is why we opt for FSDP.
"""

import shutil
from pathlib import Path
from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from prismatic.overwatch import initialize_overwatch
from prismatic.training.strategies.base_strategy import TrainingStrategy

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class DDPStrategy(TrainingStrategy):
    @overwatch.rank_zero_only
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        assert isinstance(self.vlm, DDP), "save_checkpoint assumes VLM is already wrapped in DDP!"

        # Splinter State Dictionary by Top-Level Submodules (or subset, if `only_trainable`)
        model_state_dicts = {
            mkey: getattr(self.vlm.module, mkey).state_dict()
            for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)
        }
        optimizer_state_dict = self.optimizer.state_dict()

        # Set Checkpoint Path =>> Embed *minimal* training statistics!
        checkpoint_dir = run_dir / "checkpoints"
        if train_loss is None:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
        else:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"

        # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
        torch.save({"model": model_state_dicts, "optimizer": optimizer_state_dict}, checkpoint_path)
        shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        # Gradient Checkpointing Setup
        if self.enable_gradient_checkpointing:
            # For Gradient Checkpointing --> we make the assumption that the "bulk" of activation memory is taken up
            #     by the LLM; because we also make the explicit assumption that each LLM is derived from a HF
            #     pretrained model, the only thing we *need* to do (technically) is call `gradient_checkpoint_enable`
            #     on `self.llm_backbone`.
            #
            # What does it actually do? --> runs the *generic* custom_forward + torch.utils.checkpoint.checkpoint logic
            #   => github.com/huggingface/transformers/.../models/llama/modeling_llama.py#L692-L706
            #
            # Additional Reference (to better understand gradient checkpointing in PyTorch writ large)
            #   => github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
            overwatch.info("Enabling Gradient Checkpointing on LLM Backbone", ctx_level=1)
            self.vlm.llm_backbone.gradient_checkpointing_enable()

        # Cast frozen vision backbone to half precision (same as FSDP strategy)
        if self.enable_mixed_precision_training:
            overwatch.info("Casting Vision Backbone to *Half Precision* via `.to(dtype=...)`", ctx_level=1)
            self.vlm.vision_backbone.to(dtype=self.vlm.vision_backbone.half_precision_dtype)

        # Move to Device
        overwatch.info("Placing Entire VLM (Vision Backbone, LLM Backbone, Projector Weights) on GPU", ctx_level=1)
        self.vlm.to(self.device_id)

        # Compile the LLM backbone for fused-kernel speedups (H100+)
        if self.compile_llm:
            overwatch.info("Compiling LLM Backbone with torch.compile", ctx_level=1)
            self.vlm.llm_backbone.llm = torch.compile(self.vlm.llm_backbone.llm, dynamic=True)
        else:
            overwatch.info("Skipping torch.compile for LLM Backbone", ctx_level=1)

        overwatch.info("Wrapping VLM with Distributed Data Parallel", ctx_level=1)
        self.vlm = DDP(self.vlm, device_ids=[self.device_id], gradient_as_bucket_view=True)

        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        trainable_params = [param for param in self.vlm.parameters() if param.requires_grad]
        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            if self.max_steps is None:
                num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
            else:
                num_training_steps = self.max_steps

            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        # Finalize Setup =>> Log
        overwatch.info(
            "DDP Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> LLM Backbone torch.compile = {self.compile_llm}\n"
            f"         |-> Use Native AMP = {self.enable_mixed_precision_training} ({self.mixed_precision_dtype})\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )

    def clip_grad_norm(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), max_norm=self.max_grad_norm)
