"""
Memory Calculator for PixelLM Training.

Estimates GPU memory requirements for training different variants.
"""

import argparse
from dataclasses import dataclass
from typing import Dict


@dataclass
class MemoryEstimate:
    """Memory breakdown for training."""
    model_params_gb: float
    gradients_gb: float
    optimizer_states_gb: float  # AdamW: 2x params (momentum + variance)
    activations_gb: float
    total_gb: float
    
    def __str__(self):
        return f"""
Memory Breakdown:
  Model Parameters:    {self.model_params_gb:>8.2f} GB
  Gradients:           {self.gradients_gb:>8.2f} GB
  Optimizer States:    {self.optimizer_states_gb:>8.2f} GB
  Activations:         {self.activations_gb:>8.2f} GB
  ──────────────────────────────────
  TOTAL:               {self.total_gb:>8.2f} GB
"""


# Model configurations
VARIANTS = {
    "pixel": {
        "hidden_size": 1024,
        "num_layers": 8,
        "num_heads": 8,
        "intermediate_size": 4096,
        "num_experts": 288,
        "experts_per_token": 8,
        "expert_hidden": 352,
        "vocab_size": 102400,
        "total_params_b": 6.07,
        "active_params_b": 0.339,
    },
    "megapixel": {
        "hidden_size": 2048,
        "num_layers": 16,
        "num_heads": 16,
        "intermediate_size": 8192,
        "num_experts": 288,
        "experts_per_token": 8,
        "expert_hidden": 704,
        "vocab_size": 102400,
        "total_params_b": 27.05,
        "active_params_b": 1.5,
    },
    "gigapixel": {
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "intermediate_size": 16384,
        "num_experts": 288,
        "experts_per_token": 8,
        "expert_hidden": 1408,
        "vocab_size": 102400,
        "total_params_b": 168.0,
        "active_params_b": 9.42,
    },
}

# Vision encoder (shared across variants)
VISION_PARAMS_B = 0.3  # ~300M params


def calculate_memory(
    variant: str,
    batch_size: int = 1,
    seq_length: int = 2048,
    precision: str = "bf16",
    gradient_checkpointing: bool = True,
    optimizer: str = "adamw",
    deepspeed_stage: int = 0,
) -> MemoryEstimate:
    """
    Calculate GPU memory requirements.
    
    Args:
        variant: Model variant (pixel, megapixel, gigapixel)
        batch_size: Training batch size per GPU
        seq_length: Sequence length
        precision: Training precision (fp32, bf16, fp8)
        gradient_checkpointing: Whether to use gradient checkpointing
        optimizer: Optimizer type (adamw, sgd)
        deepspeed_stage: DeepSpeed ZeRO stage (0, 1, 2, 3)
    
    Returns:
        MemoryEstimate with detailed breakdown
    """
    cfg = VARIANTS[variant]
    
    # Bytes per parameter based on precision
    bytes_per_param = {
        "fp32": 4,
        "bf16": 2,
        "fp8": 1,
    }[precision]
    
    # Total parameters (full model, not active)
    total_params = (cfg["total_params_b"] + VISION_PARAMS_B) * 1e9
    
    # Model parameters memory
    model_gb = (total_params * bytes_per_param) / (1024**3)
    
    # Gradients (same size as model in bf16/fp32)
    grad_bytes = bytes_per_param if precision != "fp8" else 2  # FP8 uses BF16 grads
    gradients_gb = (total_params * grad_bytes) / (1024**3)
    
    # Optimizer states
    if optimizer == "adamw":
        # AdamW: 2 states (momentum, variance) in FP32
        opt_states_gb = (total_params * 4 * 2) / (1024**3)
    else:
        # SGD: 1 state (momentum)
        opt_states_gb = (total_params * 4) / (1024**3)
    
    # Activations
    # Rough estimate: O(batch * seq * hidden * layers)
    hidden = cfg["hidden_size"]
    layers = cfg["num_layers"]
    
    # Base activation memory per layer
    act_per_layer = batch_size * seq_length * hidden * 4 * bytes_per_param
    
    if gradient_checkpointing:
        # Checkpointing reduces activations to sqrt(layers)
        num_stored = int(layers ** 0.5) + 1
    else:
        num_stored = layers
    
    activations_gb = (act_per_layer * num_stored) / (1024**3)
    
    # Attention: O(batch * heads * seq^2)
    attn_memory = batch_size * cfg["num_heads"] * (seq_length ** 2) * bytes_per_param
    activations_gb += (attn_memory * layers) / (1024**3)
    
    # DeepSpeed ZeRO adjustments
    if deepspeed_stage >= 1:
        # ZeRO-1: Partition optimizer states
        opt_states_gb /= 2  # Assume 2 GPUs minimum
    if deepspeed_stage >= 2:
        # ZeRO-2: Partition gradients
        gradients_gb /= 2
    if deepspeed_stage >= 3:
        # ZeRO-3: Partition parameters
        model_gb /= 2
    
    total = model_gb + gradients_gb + opt_states_gb + activations_gb
    
    return MemoryEstimate(
        model_params_gb=model_gb,
        gradients_gb=gradients_gb,
        optimizer_states_gb=opt_states_gb,
        activations_gb=activations_gb,
        total_gb=total,
    )


def main():
    parser = argparse.ArgumentParser(description="PixelLM Memory Calculator")
    parser.add_argument("--variant", type=str, default="pixel", choices=["pixel", "megapixel", "gigapixel"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp8"])
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--deepspeed-stage", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--gpu-memory", type=float, default=96, help="GPU memory in GB")
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"PixelLM Memory Estimation: {args.variant.upper()}")
    print("=" * 60)
    
    cfg = VARIANTS[args.variant]
    print(f"\nModel Configuration:")
    print(f"  Total Parameters:  {cfg['total_params_b']:.2f}B (LLM) + {VISION_PARAMS_B}B (Vision)")
    print(f"  Active Parameters: {cfg['active_params_b']:.2f}B")
    print(f"  Layers:            {cfg['num_layers']}")
    print(f"  Hidden Size:       {cfg['hidden_size']}")
    print(f"  Experts:           {cfg['num_experts']} (top-{cfg['experts_per_token']})")
    
    print(f"\nTraining Configuration:")
    print(f"  Batch Size:        {args.batch_size} per GPU")
    print(f"  Sequence Length:   {args.seq_length}")
    print(f"  Precision:         {args.precision}")
    print(f"  Gradient Ckpt:     {not args.no_checkpointing}")
    print(f"  DeepSpeed Stage:   {args.deepspeed_stage}")
    
    estimate = calculate_memory(
        variant=args.variant,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        precision=args.precision,
        gradient_checkpointing=not args.no_checkpointing,
        deepspeed_stage=args.deepspeed_stage,
    )
    
    print(estimate)
    
    total_vram = args.num_gpus * args.gpu_memory
    per_gpu = estimate.total_gb / args.num_gpus if args.num_gpus > 1 else estimate.total_gb
    
    print(f"GPU Configuration:")
    print(f"  Number of GPUs:    {args.num_gpus}")
    print(f"  VRAM per GPU:      {args.gpu_memory:.0f} GB")
    print(f"  Total VRAM:        {total_vram:.0f} GB")
    print(f"  Required per GPU:  {per_gpu:.2f} GB")
    
    print()
    if per_gpu <= args.gpu_memory * 0.9:  # 90% threshold for safety
        print(f"✅ SUFFICIENT: {per_gpu:.1f} GB required < {args.gpu_memory * 0.9:.1f} GB available")
    elif per_gpu <= args.gpu_memory:
        print(f"⚠️  TIGHT: {per_gpu:.1f} GB required ≈ {args.gpu_memory:.1f} GB available")
        print("   → Consider reducing batch size or using DeepSpeed ZeRO-3")
    else:
        print(f"❌ INSUFFICIENT: {per_gpu:.1f} GB required > {args.gpu_memory:.1f} GB available")
        print("   → Use DeepSpeed ZeRO-3 with CPU offloading")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("Recommendations for 2x RTX PRO 6000 (192 GB total)")
    print("=" * 60)
    
    for v in ["pixel", "megapixel", "gigapixel"]:
        for ds in [0, 3]:
            est = calculate_memory(v, batch_size=1, deepspeed_stage=ds)
            per = est.total_gb / 2
            status = "✅" if per < 96 else "❌"
            ds_str = f"ZeRO-{ds}" if ds > 0 else "No ZeRO"
            print(f"  {v:12} + {ds_str:8}: {per:6.1f} GB/GPU {status}")


if __name__ == "__main__":
    main()
