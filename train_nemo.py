"""
NeMo Training Script for PixelLM.

Uses NeMo 2.0 with Megatron Core for:
  - Stage 1: Vision-Language Alignment
  - Stage 2: Pretraining
  - Stage 3: Supervised Fine-Tuning

Supports 3D parallelism (TP + PP + DP) and Expert Parallelism for MoE.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

# NeMo imports
try:
    import nemo_run as run
    from nemo import lightning as nl
    from nemo.collections.llm import PreTrainingDataModule
    from nemo.collections.vlm import (
        NevaModel,
        NevaConfig,
        ImageDataConfig,
    )
    from nemo.lightning import MegatronStrategy, NeMoLogger
    from nemo.lightning.pytorch.callbacks import ModelCheckpoint
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print("Warning: NeMo not installed. Install with: pip install nemo_toolkit[all]")

import torch


# =============================================================================
# Model Configuration
# =============================================================================

def get_neva_config(variant: str) -> dict:
    """Get NeMo NeVA config for PixelLM variant."""
    
    configs = {
        "pixel": {
            "hidden_size": 1024,
            "num_layers": 8,
            "num_attention_heads": 8,
            "ffn_hidden_size": 4096,
            "num_moe_experts": 288,
            "moe_router_topk": 8,
        },
        "megapixel": {
            "hidden_size": 2048,
            "num_layers": 16,
            "num_attention_heads": 16,
            "ffn_hidden_size": 8192,
            "num_moe_experts": 288,
            "moe_router_topk": 8,
        },
        "gigapixel": {
            "hidden_size": 4096,
            "num_layers": 32,
            "num_attention_heads": 32,
            "ffn_hidden_size": 16384,
            "num_moe_experts": 288,
            "moe_router_topk": 8,
        },
    }
    
    return configs.get(variant, configs["pixel"])


# =============================================================================
# Data Configuration
# =============================================================================

STAGE_DATA_CONFIGS = {
    "stage1": {
        "datasets": ["liuhaotian/LLaVA-CC3M-Pretrain-595K"],
        "max_seq_length": 2048,
        "micro_batch_size": 4,
        "global_batch_size": 256,
    },
    "stage2": {
        "datasets": ["laion/laion400m", "kakaobrain/coyo-700m"],
        "text_datasets": ["HuggingFaceFW/fineweb"],
        "text_ratio": 0.3,
        "max_seq_length": 4096,
        "micro_batch_size": 2,
        "global_batch_size": 512,
    },
    "stage3": {
        "datasets": ["liuhaotian/LLaVA-Instruct-150K"],
        "max_seq_length": 4096,
        "micro_batch_size": 2,
        "global_batch_size": 128,
    },
}


# =============================================================================
# Parallelism Configuration
# =============================================================================

def get_parallelism_config(variant: str, num_gpus: int) -> dict:
    """Get Megatron 3D parallelism config based on variant and GPU count."""
    
    if variant == "gigapixel":
        # Large model: aggressive parallelism
        return {
            "tensor_model_parallel_size": min(8, num_gpus),
            "pipeline_model_parallel_size": min(4, num_gpus // 8) if num_gpus >= 8 else 1,
            "expert_model_parallel_size": min(8, num_gpus),
            "context_parallel_size": 1,
            "sequence_parallel": True,
        }
    elif variant == "megapixel":
        # Medium model
        return {
            "tensor_model_parallel_size": min(4, num_gpus),
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": min(4, num_gpus),
            "context_parallel_size": 1,
            "sequence_parallel": True,
        }
    else:
        # Pixel: smaller model
        return {
            "tensor_model_parallel_size": min(2, num_gpus),
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "context_parallel_size": 1,
            "sequence_parallel": False,
        }


# =============================================================================
# Training Recipe
# =============================================================================

def create_training_recipe(
    stage: str,
    variant: str,
    output_dir: str,
    resume_from: Optional[str] = None,
    num_gpus: int = 1,
    num_nodes: int = 1,
    max_steps: int = 10000,
):
    """Create NeMo training recipe."""
    
    if not NEMO_AVAILABLE:
        raise RuntimeError("NeMo not installed. Install with: pip install nemo_toolkit[all]")
    
    # Get configs
    model_cfg = get_neva_config(variant)
    data_cfg = STAGE_DATA_CONFIGS[stage]
    parallel_cfg = get_parallelism_config(variant, num_gpus * num_nodes)
    
    # Model
    model = run.Config(
        NevaModel,
        config=run.Config(
            NevaConfig,
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["num_layers"],
            num_attention_heads=model_cfg["num_attention_heads"],
            ffn_hidden_size=model_cfg["ffn_hidden_size"],
            num_moe_experts=model_cfg["num_moe_experts"],
            moe_router_topk=model_cfg["moe_router_topk"],
            # Vision encoder
            vision_model_type="siglip",
            vision_hidden_size=1024,
            vision_num_layers=24,
            vision_patch_size=16,
            vision_image_size=384,
        ),
    )
    
    # Data
    data = run.Config(
        PreTrainingDataModule,
        paths=data_cfg["datasets"],
        seq_length=data_cfg["max_seq_length"],
        micro_batch_size=data_cfg["micro_batch_size"],
        global_batch_size=data_cfg["global_batch_size"],
    )
    
    # Strategy (Megatron 3D parallelism)
    strategy = run.Config(
        MegatronStrategy,
        tensor_model_parallel_size=parallel_cfg["tensor_model_parallel_size"],
        pipeline_model_parallel_size=parallel_cfg["pipeline_model_parallel_size"],
        expert_model_parallel_size=parallel_cfg["expert_model_parallel_size"],
        context_parallel_size=parallel_cfg["context_parallel_size"],
        sequence_parallel=parallel_cfg["sequence_parallel"],
        gradient_as_bucket_view=True,
        ckpt_async_save=True,
    )
    
    # Trainer
    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        devices=num_gpus,
        num_nodes=num_nodes,
        max_steps=max_steps,
        strategy=strategy,
        precision="bf16-mixed",
        log_every_n_steps=10,
        val_check_interval=1000,
        callbacks=[
            run.Config(
                ModelCheckpoint,
                dirpath=f"{output_dir}/checkpoints",
                save_top_k=3,
                monitor="val_loss",
                mode="min",
            ),
        ],
    )
    
    # Logger
    logger = run.Config(
        NeMoLogger,
        log_dir=output_dir,
        name=f"pixellm_{variant}_{stage}",
        use_datetime_version=True,
        tensorboard=True,
        wandb=None,  # Set to wandb config if needed
    )
    
    # Resume
    resume = None
    if resume_from:
        resume = run.Config(
            nl.AutoResume,
            restore_config=run.Config(
                nl.RestoreConfig,
                path=resume_from,
            ),
        )
    
    return run.Partial(
        nl.pretrain,
        model=model,
        data=data,
        trainer=trainer,
        log=logger,
        resume=resume,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="NeMo Training for PixelLM")
    parser.add_argument("--stage", type=str, required=True, choices=["stage1", "stage2", "stage3"])
    parser.add_argument("--variant", type=str, default="pixel", choices=["pixel", "megapixel", "gigapixel"])
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10000)
    args = parser.parse_args()
    
    if not NEMO_AVAILABLE:
        print("Error: NeMo not installed.")
        print("Install with: pip install nemo_toolkit[all]")
        return
    
    print(f"Creating training recipe for {args.variant} - {args.stage}")
    print(f"GPUs: {args.num_gpus} x {args.num_nodes} nodes")
    
    recipe = create_training_recipe(
        stage=args.stage,
        variant=args.variant,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        num_gpus=args.num_gpus,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
    )
    
    # Execute
    run.run(recipe, executor=run.LocalExecutor())


if __name__ == "__main__":
    main()
