"""
Training Optimizations for PixelLM.

Critical features:
  - Data Packing: Pack multiple samples to minimize padding waste
  - EMA: Exponential Moving Average for stable training
  - Per-module LR: Different learning rates for vision/language
  - Image Augmentation: Transforms for robustness
"""

import copy
import math
from typing import Dict, List, Optional, Iterator, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


# =============================================================================
# Data Packing (Critical for efficiency)
# =============================================================================

class PackedDataset(IterableDataset):
    """
    Pack multiple samples into single sequences to minimize padding waste.
    
    Without packing: [sample1][PAD][PAD][PAD] - 50%+ waste
    With packing:    [sample1][sample2][sample3] - ~0% waste
    
    Industry standard for pretraining efficiency.
    """
    
    def __init__(
        self,
        dataset: IterableDataset,
        max_seq_len: int = 8192,
        pad_token_id: int = 0,
        eos_token_id: int = 2,
    ):
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer_ids = []
        buffer_labels = []
        buffer_images = []
        buffer_positions = []
        
        for sample in self.dataset:
            ids = sample.get("input_ids")
            labels = sample.get("labels", ids.clone() if isinstance(ids, torch.Tensor) else ids)
            
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()
            
            # Add to buffer
            buffer_ids.extend(ids)
            buffer_labels.extend(labels)
            
            # Track image positions
            if "images" in sample and sample["images"] is not None:
                buffer_images.append(sample["images"])
                buffer_positions.append(len(buffer_ids) - len(ids))
            
            # Emit packed sequence when buffer is full
            while len(buffer_ids) >= self.max_seq_len:
                packed_ids = buffer_ids[:self.max_seq_len]
                packed_labels = buffer_labels[:self.max_seq_len]
                
                buffer_ids = buffer_ids[self.max_seq_len:]
                buffer_labels = buffer_labels[self.max_seq_len:]
                
                yield {
                    "input_ids": torch.tensor(packed_ids, dtype=torch.long),
                    "labels": torch.tensor(packed_labels, dtype=torch.long),
                    "attention_mask": torch.ones(self.max_seq_len, dtype=torch.long),
                    "images": torch.stack(buffer_images) if buffer_images else None,
                    "image_positions": torch.tensor(buffer_positions) if buffer_positions else None,
                }
                
                # Reset image buffers after emit
                buffer_images = []
                buffer_positions = []


# =============================================================================
# EMA (Exponential Moving Average)
# =============================================================================

class EMAModel:
    """
    Exponential Moving Average of model parameters.
    
    Maintains a shadow copy of model weights that's smoother and often
    performs better at inference time.
    
    Industry standard for stable training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_after_step: int = 100,
        update_every: int = 10,
    ):
        self.decay = decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.step = 0
        
        # Create shadow parameters
        self.shadow_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """Update EMA parameters."""
        self.step += 1
        
        if self.step < self.update_after_step:
            return
        
        if self.step % self.update_every != 0:
            return
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow_params:
                    self.shadow_params[name].lerp_(param.data, 1 - self.decay)
    
    def apply_to(self, model: nn.Module):
        """Apply EMA weights to model."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow_params:
                    param.data.copy_(self.shadow_params[name])
    
    def store(self, model: nn.Module):
        """Store current model weights."""
        self.stored_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.stored_params[name] = param.data.clone()
    
    def restore(self, model: nn.Module):
        """Restore stored weights."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.stored_params:
                    param.data.copy_(self.stored_params[name])
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": self.decay,
            "step": self.step,
            "shadow_params": self.shadow_params,
        }
    
    def load_state_dict(self, state: Dict[str, Any]):
        self.decay = state["decay"]
        self.step = state["step"]
        self.shadow_params = state["shadow_params"]


# =============================================================================
# Per-Module Learning Rates
# =============================================================================

@dataclass
class ModuleLRConfig:
    """Learning rate configuration per module."""
    vision_lr: float = 1e-5      # Lower LR for pretrained vision
    projector_lr: float = 1e-4   # Medium LR for projector
    language_lr: float = 1e-4    # Standard LR for language
    moe_lr: float = 1e-4         # MoE experts
    

def get_param_groups(
    model: nn.Module,
    config: ModuleLRConfig,
    weight_decay: float = 0.1,
) -> List[Dict]:
    """
    Create parameter groups with different learning rates.
    
    Industry standard: lower LR for pretrained vision encoder.
    """
    
    vision_params = []
    projector_params = []
    language_params = []
    moe_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No decay for biases and LayerNorm
        if "bias" in name or "norm" in name or "embed" in name:
            no_decay_params.append(param)
        elif "vision" in name:
            vision_params.append(param)
        elif "projector" in name:
            projector_params.append(param)
        elif "expert" in name or "moe" in name:
            moe_params.append(param)
        else:
            language_params.append(param)
    
    param_groups = [
        {"params": vision_params, "lr": config.vision_lr, "weight_decay": weight_decay, "name": "vision"},
        {"params": projector_params, "lr": config.projector_lr, "weight_decay": weight_decay, "name": "projector"},
        {"params": language_params, "lr": config.language_lr, "weight_decay": weight_decay, "name": "language"},
        {"params": moe_params, "lr": config.moe_lr, "weight_decay": weight_decay, "name": "moe"},
        {"params": no_decay_params, "lr": config.language_lr, "weight_decay": 0.0, "name": "no_decay"},
    ]
    
    # Filter empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    
    return param_groups


# =============================================================================
# Image Augmentation
# =============================================================================

class ImageAugmentation:
    """
    Image augmentation for robust VLM training.
    
    Standard transforms used in vision pretraining.
    """
    
    def __init__(
        self,
        image_size: int = 384,
        scale: tuple = (0.8, 1.0),
        ratio: tuple = (0.9, 1.1),
        hflip_prob: float = 0.5,
        color_jitter: float = 0.1,
        enabled: bool = True,
    ):
        self.image_size = image_size
        self.scale = scale
        self.ratio = ratio
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter
        self.enabled = enabled
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to image tensor (C, H, W)."""
        
        if not self.enabled:
            return image
        
        import torch.nn.functional as F
        import random
        
        C, H, W = image.shape
        
        # Random horizontal flip
        if random.random() < self.hflip_prob:
            image = image.flip(-1)
        
        # Random resized crop
        if random.random() < 0.5:
            scale = random.uniform(*self.scale)
            new_h = int(H * scale)
            new_w = int(W * scale)
            
            top = random.randint(0, H - new_h) if H > new_h else 0
            left = random.randint(0, W - new_w) if W > new_w else 0
            
            image = image[:, top:top+new_h, left:left+new_w]
            image = F.interpolate(
                image.unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # Color jitter (brightness, contrast)
        if self.color_jitter > 0 and random.random() < 0.5:
            brightness = 1 + random.uniform(-self.color_jitter, self.color_jitter)
            image = image * brightness
            image = image.clamp(0, 1)
        
        return image


# =============================================================================
# Training Callback for EMA
# =============================================================================

class EMACallback:
    """
    Callback to update EMA during training.
    
    Usage with HuggingFace Trainer:
        ema = EMAModel(model)
        trainer.add_callback(EMACallback(ema))
    """
    
    def __init__(self, ema: EMAModel):
        self.ema = ema
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self.ema.update(model)
    
    def on_save(self, args, state, control, model=None, **kwargs):
        if model is not None:
            # Save EMA state
            import os
            ema_path = os.path.join(args.output_dir, f"ema-{state.global_step}.pt")
            torch.save(self.ema.state_dict(), ema_path)


# =============================================================================
# Optimized Data Collator
# =============================================================================

@dataclass
class OptimizedVLCollator:
    """
    Optimized data collator with packing support.
    """
    
    pad_token_id: int = 0
    max_length: int = 8192
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        
        # Stack tensors
        for key in ['input_ids', 'labels', 'attention_mask']:
            if key in features[0] and features[0][key] is not None:
                tensors = [f[key] for f in features if f.get(key) is not None]
                if tensors:
                    batch[key] = torch.stack(tensors)
        
        # Handle images
        images = [f['images'] for f in features if f.get('images') is not None]
        if images:
            # Flatten if nested
            flat_images = []
            for img in images:
                if img.dim() == 4:  # (N, C, H, W)
                    flat_images.extend([img[i] for i in range(img.shape[0])])
                else:
                    flat_images.append(img)
            if flat_images:
                batch['images'] = torch.stack(flat_images)
        
        # Image positions
        positions = [f['image_positions'] for f in features if f.get('image_positions') is not None]
        if positions:
            batch['image_positions'] = torch.cat(positions)
        
        # DPO fields
        for key in ['chosen_input_ids', 'chosen_labels', 'rejected_input_ids', 'rejected_labels']:
            if key in features[0] and features[0][key] is not None:
                batch[key] = torch.stack([f[key] for f in features])
        
        return batch


# Export
__all__ = [
    "PackedDataset",
    "EMAModel",
    "EMACallback",
    "ModuleLRConfig",
    "get_param_groups",
    "ImageAugmentation",
    "OptimizedVLCollator",
]
