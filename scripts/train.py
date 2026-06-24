"""
train.py

Single-stage VLM finetuning: salaadpp LLM checkpoint + frozen vision encoder →
randomly-initialized projector, trained end-to-end (vision frozen).

The LLM is loaded from a salaadpp resume.pth (same format as garage_core /
garage_eval). model_config.json is auto-detected from the same directory.

Run with:
    torchrun --standalone --nnodes 1 --nproc-per-node $K scripts/train.py \
        llm=/fast/txia/salaadpp/1b_vanilla/resume.pth

    # With SALAD-compressed LLM:
    torchrun --standalone --nnodes 1 --nproc-per-node $K scripts/train.py \
        llm=/fast/txia/salaadpp/1b_no_s_da0.03/resume.pth \
        artifact=/path/to/artifact.pt
"""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Type

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from prismatic.models import get_vision_backbone_and_transform, get_vlm
from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, PromptBuilder
from prismatic.models.lowrank_linear import LowRankLinear, LoRALinear
from prismatic.overwatch import initialize_overwatch
from prismatic.training import Metrics, get_train_strategy
from prismatic.util import set_global_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.set_float32_matmul_precision("high")

OmegaConf.register_new_resolver("parent_name", lambda p: Path(p).parent.name, replace=True)

overwatch = initialize_overwatch(__name__)


# ---------------------------------------------------------------------------
# SalaadLLMBackbone: wraps a salaadpp resume.pth into prismatic's LLMBackbone
# ---------------------------------------------------------------------------

def _read_model_config(path: Path) -> dict:
    return json.loads(path.read_text())


def _build_llama(cfg_dict: dict) -> LlamaForCausalLM:
    """Instantiate a fresh LlamaForCausalLM from a salaadpp model_config.json."""
    if "max_sequence_length" in cfg_dict and "max_position_embeddings" not in cfg_dict:
        cfg_dict["max_position_embeddings"] = cfg_dict.pop("max_sequence_length")
    if cfg_dict.get("pad_token_id") == -1:
        cfg_dict["pad_token_id"] = None
    cfg_dict["use_cache"] = False
    return LlamaForCausalLM(LlamaConfig(**cfg_dict))


def _load_state_dict(path: Path) -> dict:
    raw = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    sd = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}
    return sd


class SalaadLLMBackbone(LLMBackbone):
    """LLMBackbone loaded from a salaadpp resume.pth + model_config.json."""

    def __init__(
        self,
        resume_path: Path,
        model_config_path: Path,
        llm_max_length: int = 2048,
    ) -> None:
        super().__init__(llm_backbone_id=resume_path.parent.name)

        cfg_dict = _read_model_config(model_config_path)
        tokenizer_id = cfg_dict.pop("tokenizer_id", "t5-base")

        overwatch.info(
            f"Building LLM from {model_config_path.name} "
            f"(hidden={cfg_dict.get('hidden_size')}, layers={cfg_dict.get('num_hidden_layers')})"
        )
        self.llm = _build_llama(cfg_dict)

        overwatch.info(f"Loading LLM weights from {resume_path}")
        self.llm.load_state_dict(_load_state_dict(resume_path))

        self.llm.enable_input_require_grads()

        overwatch.info(f"Loading tokenizer: {tokenizer_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            model_max_length=llm_max_length,
            padding_side="right",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
            self.llm.config.pad_token_id = self.tokenizer.pad_token_id
            self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return PurePromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def get_fsdp_wrapping_policy(self) -> Callable:
        return partial(transformer_auto_wrap_policy, transformer_layer_cls={self.transformer_layer_cls})

    def enable_gradient_checkpointing(self) -> None:
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    gradient_checkpointing_enable = enable_gradient_checkpointing

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


# ---------------------------------------------------------------------------
# SALAD artifact application
# ---------------------------------------------------------------------------

def _apply_artifact(llm: nn.Module, artifact_path: Path) -> int:
    """Replace tracked nn.Linear layers in `llm` with LowRankLinear from artifact."""
    raw = torch.load(str(artifact_path), map_location="cpu", weights_only=False)
    layers = raw["layers"]
    meta = raw.get("meta", {})
    overwatch.info(
        f"Loaded artifact: {len(layers)} layers, "
        f"mode={meta.get('mode', '?')}, params={meta.get('param_count', '?')}"
    )

    n_replaced = 0
    for name, layer_data in layers.items():
        if "W" in layer_data:
            continue

        try:
            old = llm.get_submodule(name)
        except AttributeError:
            overwatch.warning(f"Artifact key {name!r} not found in LLM — skipping")
            continue

        if isinstance(old, nn.Embedding):
            A, B = layer_data["A"].float(), layer_data["B"].float()
            old.weight.data.copy_((A @ B).to(old.weight.dtype))
            n_replaced += 1
            continue

        if not isinstance(old, nn.Linear):
            overwatch.warning(f"Expected nn.Linear at {name}, got {type(old).__name__} — skipping")
            continue

        new = LowRankLinear(
            A=layer_data["A"],
            B=layer_data["B"],
            bias=old.bias.data if old.bias is not None else None,
        )

        parent_name, _, attr = name.rpartition(".")
        parent = llm.get_submodule(parent_name) if parent_name else llm
        setattr(parent, attr, new)
        n_replaced += 1

    overwatch.info(f"Replaced {n_replaced} layers with LowRankLinear")
    return n_replaced


# ---------------------------------------------------------------------------
# LoRA application
# ---------------------------------------------------------------------------

LORA_TARGET_PRESETS = {
    "qv": {"q_proj", "v_proj"},
    "attn": {"q_proj", "k_proj", "v_proj", "o_proj"},
    "all": {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
}


def _resolve_lora_targets(targets: str) -> set[str]:
    if targets in LORA_TARGET_PRESETS:
        return LORA_TARGET_PRESETS[targets]

    resolved = {target.strip() for target in targets.split(",") if target.strip()}
    if not resolved:
        raise ValueError("`lora_targets` must be one of qv|attn|all or a comma-separated target list")
    return resolved


def _apply_lora(llm: nn.Module, r: int, alpha: float, targets: set[str]) -> int:
    """Wrap target nn.Linear layers in `llm` with LoRALinear (frozen base + trainable adapters)."""
    n_applied = 0
    for name, module in list(llm.named_modules()):
        if not isinstance(module, (nn.Linear, LowRankLinear)):
            continue
        short_name = name.rsplit(".", 1)[-1]
        if short_name not in targets:
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = llm.get_submodule(parent_name) if parent_name else llm
        setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha))
        n_applied += 1
    lora_params = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in llm.parameters())
    overwatch.info(
        f"Applied LoRA (r={r}, alpha={alpha}) to {n_applied} layers — "
        f"trainable: {lora_params:,} / {total_params:,} ({100 * lora_params / total_params:.1f}%)"
    )
    return n_applied


# ---------------------------------------------------------------------------
# Auto-detect model_config.json
# ---------------------------------------------------------------------------

def _find_model_config(resume_path: Path) -> Path:
    """Look for model_config.json in the same directory as resume.pth."""
    candidates = list(resume_path.parent.glob("*model_config*.json")) + \
                 list(resume_path.parent.glob("model_config.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No model_config.json found in {resume_path.parent}. "
            f"Expected a JSON file matching *model_config*.json alongside {resume_path.name}."
        )
    if len(set(candidates)) > 1:
        overwatch.warning(f"Multiple model configs found: {candidates}; using {candidates[0]}")
    return candidates[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_VISION = "dinosiglip-vit-so-384px"
DEFAULT_RESIZE = "resize-naive"
DEFAULT_ARCH = "no-align+fused-gelu-mlp"


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    overwatch.info("Prismatic VLM Training (single-stage, salaadpp LLM)")

    resume_path = Path(cfg.llm)
    if not resume_path.exists():
        raise FileNotFoundError(f"LLM checkpoint not found: {resume_path}")

    model_config_path = _find_model_config(resume_path)
    overwatch.info(f"LLM checkpoint : {resume_path}")
    overwatch.info(f"Model config   : {model_config_path}")

    vision_id = cfg.get("vision", DEFAULT_VISION)
    resize_strategy = cfg.get("resize_strategy", DEFAULT_RESIZE)
    arch_specifier = cfg.get("arch_specifier", DEFAULT_ARCH)

    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)

    from hydra.core.hydra_config import HydraConfig
    run_dir = Path(HydraConfig.get().run.dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Build components (rank 0 downloads weights/packages first, others read from cache) ---
    with overwatch.local_zero_first():
        vision_backbone, image_transform = get_vision_backbone_and_transform(vision_id, resize_strategy)
        llm_backbone = SalaadLLMBackbone(
            resume_path,
            model_config_path,
            llm_max_length=cfg.get("llm_max_length", 2048),
        )
    overwatch.info(f"Vision backbone: {vision_id} ({resize_strategy})")
    tokenizer = llm_backbone.get_tokenizer()

    model_id = f"salaad-{resume_path.parent.name}+{vision_id}"
    vlm = get_vlm(
        model_id,
        arch_specifier,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=(cfg.trainer.precision == "bf16"),
    )

    vlm.freeze_backbones("finetune")

    # --- Optional: apply SALAD artifact (compressed LLM) ---
    artifact_path = cfg.get("artifact")
    if artifact_path is not None:
        overwatch.info(f"Applying SALAD artifact from {artifact_path}")
        _apply_artifact(vlm.llm_backbone.llm, Path(artifact_path))

    # --- Optional: LoRA (freeze LLM base, only train adapters + projector) ---
    lora_r = cfg.get("lora_r")
    if lora_r is not None:
        lora_alpha = cfg.get("lora_alpha") or float(lora_r)
        lora_targets = _resolve_lora_targets(cfg.get("lora_targets", "all"))
        vlm.llm_backbone.requires_grad_(False)
        _apply_lora(vlm.llm_backbone.llm, r=int(lora_r), alpha=float(lora_alpha), targets=lora_targets)
        vlm.projector.requires_grad_(True)

    # --- Save run config ---
    if overwatch.is_rank_zero():
        run_cfg = {
            "llm": str(resume_path),
            "model_config": str(model_config_path),
            "vision": vision_id,
            "resize_strategy": resize_strategy,
            "arch_specifier": arch_specifier,
            "artifact": str(artifact_path) if artifact_path else None,
            "lora_r": int(lora_r) if lora_r is not None else None,
            "lora_alpha": float(lora_alpha) if lora_r is not None else None,
            "train": OmegaConf.to_container(cfg, resolve=True),
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(run_cfg, f, indent=2)

    # --- Dataset ---
    data_root = Path(cfg.data.dataset_root_dir)
    annotation_json = data_root / cfg.data.finetune_json
    image_dir = data_root / cfg.data.finetune_image_dir

    from prismatic.preprocessing.datasets import FinetuneDataset
    from prismatic.util.data_utils import PaddedCollatorForLanguageModeling

    train_dataset = FinetuneDataset(
        annotation_json, image_dir, image_transform, tokenizer,
        prompt_builder_fn=llm_backbone.prompt_builder_fn,
    )
    collator = PaddedCollatorForLanguageModeling(
        tokenizer.model_max_length,
        tokenizer.pad_token_id,
        vision_backbone.default_image_resolution,
        padding_side=tokenizer.padding_side,
    )

    # --- Training strategy (FSDP) ---
    train_strategy = get_train_strategy(
        train_strategy=cfg.trainer.train_strategy,
        vlm=vlm,
        device_id=device_id,
        epochs=cfg.schedule.epochs,
        max_steps=cfg.schedule.max_steps,
        global_batch_size=cfg.trainer.global_batch_size,
        per_device_batch_size=cfg.trainer.per_device_batch_size,
        learning_rate=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        max_grad_norm=cfg.trainer.grad_clip,
        lr_scheduler_type="linear-warmup+cosine-decay",
        warmup_ratio=cfg.schedule.warmup_ratio,
        enable_gradient_checkpointing=cfg.trainer.gradient_checkpointing,
        compile_llm=cfg.trainer.compile_llm,
        enable_mixed_precision_training=(cfg.trainer.precision == "bf16"),
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(train_dataset))

    # --- Metrics ---
    metrics = Metrics(
        active_trackers=("jsonl", "wandb"),
        run_id=run_dir.name,
        run_dir=run_dir,
        hparams=OmegaConf.to_container(cfg, resolve=True),
        stage="finetune",
        wandb_project=cfg.get("wandb_project", "salaad-vlm"),
        wandb_entity=cfg.get("wandb_entity", None),
        grad_accumulation_steps=train_strategy.grad_accumulation_steps,
    )

    # --- Train ---
    overwatch.info("Starting finetuning")
    train_strategy.run_training(train_dataset, collator, metrics, stage="finetune", seed=cfg.seed)

    metrics.finalize()
    overwatch.info("Done")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
