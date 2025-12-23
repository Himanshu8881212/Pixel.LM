"""
FP8 Training Utilities for PixelLM.

Provides FP8 support via NVIDIA TransformerEngine for Hopper GPUs (H100/H200).
Falls back to BF16 on unsupported hardware.
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

# Try importing TransformerEngine
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    TE_AVAILABLE = True
except ImportError:
    TE_AVAILABLE = False

# Try importing Flash Attention 3
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.flash_attn_interface import flash_attn_with_kvcache
    FLASH_ATTN_AVAILABLE = True
    try:
        # Check for FA3 specific features
        from flash_attn import __version__ as fa_version
        FLASH_ATTN_3 = int(fa_version.split('.')[0]) >= 3
    except:
        FLASH_ATTN_3 = False
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    FLASH_ATTN_3 = False


@dataclass
class FP8Config:
    """Configuration for FP8 training."""
    enabled: bool = True
    margin: int = 0
    interval: int = 1
    fp8_format: str = "HYBRID"  # HYBRID or E4M3
    amax_history_len: int = 1024
    amax_compute_algo: str = "max"
    
    def get_recipe(self):
        """Get TransformerEngine DelayedScaling recipe."""
        if not TE_AVAILABLE:
            return None
        return DelayedScaling(
            margin=self.margin,
            interval=self.interval,
            fp8_format=Format.HYBRID if self.fp8_format == "HYBRID" else Format.E4M3,
            amax_history_len=self.amax_history_len,
            amax_compute_algo=self.amax_compute_algo,
        )


class FP8Linear(nn.Module):
    """FP8-enabled Linear layer using TransformerEngine."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        if TE_AVAILABLE:
            self.linear = te.Linear(in_features, out_features, bias=bias)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FP8LayerNorm(nn.Module):
    """FP8-enabled LayerNorm using TransformerEngine."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        if TE_AVAILABLE:
            self.norm = te.LayerNorm(hidden_size, eps=eps)
        else:
            self.norm = nn.LayerNorm(hidden_size, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


def fp8_autocast(enabled: bool = True, fp8_recipe = None):
    """Context manager for FP8 training.
    
    Args:
        enabled: Whether to enable FP8
        fp8_recipe: TransformerEngine DelayedScaling recipe
        
    Usage:
        with fp8_autocast(enabled=True, fp8_recipe=config.get_recipe()):
            output = model(input)
    """
    if TE_AVAILABLE and enabled:
        return te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
    else:
        # Fallback to no-op context manager
        import contextlib
        return contextlib.nullcontext()


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    causal: bool = True,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Flash Attention wrapper with FA3 support.
    
    Args:
        q: Query tensor (B, H, L, D)
        k: Key tensor (B, H, L, D)
        v: Value tensor (B, H, L, D)
        causal: Use causal masking
        dropout_p: Dropout probability
    
    Returns:
        Attention output (B, H, L, D)
    """
    if FLASH_ATTN_AVAILABLE:
        # Flash Attention expects (B, L, H, D) format
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out = flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)
        return out.transpose(1, 2)
    else:
        # Fallback to PyTorch SDPA
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal, dropout_p=dropout_p
        )


def get_device_capability() -> tuple:
    """Get CUDA device compute capability."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()
    return (0, 0)


def supports_fp8() -> bool:
    """Check if current device supports FP8 (Hopper/Ada)."""
    major, _ = get_device_capability()
    return major >= 9  # Hopper is compute capability 9.0


def supports_flash_attn_3() -> bool:
    """Check if Flash Attention 3 is available."""
    return FLASH_ATTN_3 and supports_fp8()


# Export availability flags
__all__ = [
    "FP8Config",
    "FP8Linear", 
    "FP8LayerNorm",
    "fp8_autocast",
    "flash_attention",
    "supports_fp8",
    "supports_flash_attn_3",
    "TE_AVAILABLE",
    "FLASH_ATTN_AVAILABLE",
    "FLASH_ATTN_3",
]
