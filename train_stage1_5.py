"""
Stage 1.5: Vision Encoder Incremental Learning (InternVL2 style).

Unfreezes the Vision Transformer (ViT) to enhance visual understanding:
  - Improves OCR, chart, and document understanding
  - Uses smaller LLM (frozen) to reduce compute
  - Continues from Stage 1 checkpoint
"""

import argparse
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from transformers import Trainer, TrainingArguments
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@dataclass
class Stage1_5Config:
    """Configuration for ViT incremental learning."""
    
    # Model
    model_path: str = "./outputs/stage1/checkpoints/latest"
    variant: str = "pixel"
    
    # What to train
    train_vision: bool = True
    train_projector: bool = True
    train_language: bool = False  # Keep LLM frozen
    
    # Data (focus on domains less represented in web data)
    datasets: list = None
    
    # Training
    learning_rate: float = 1e-5  # Lower LR for fine-tuning
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_steps: int = 5000
    warmup_steps: int = 200
    
    # Output
    output_dir: str = "./outputs/stage1.5"
    
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = [
                # OCR and document understanding
                "naver-clova-ix/synthdog-en",
                "nielsr/docvqa_1200_examples",
                # Chart and diagram understanding
                "ahmed-masry/ChartQA",
                # Mathematical content
                "AI4Math/MathVista",
            ]


class ViTUnfreezeTrainer:
    """
    Trainer for ViT incremental learning.
    
    Following InternVL2 approach:
      1. Unfreeze vision encoder
      2. Keep LLM frozen to reduce compute
      3. Train on specialized domains
    """
    
    def __init__(self, config: Stage1_5Config):
        self.config = config
        self.model = None
        self.optimizer = None
    
    def load_model(self):
        """Load model from Stage 1 checkpoint."""
        from src.model import PixelLM, get_config
        
        cfg = get_config(self.config.variant)
        self.model = PixelLM(cfg)
        
        # Load checkpoint
        if os.path.exists(self.config.model_path):
            state_dict = torch.load(
                f"{self.config.model_path}/pytorch_model.bin",
                map_location="cpu"
            )
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {self.config.model_path}")
    
    def configure_trainable_params(self):
        """Set which parameters to train."""
        
        # Freeze everything first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze vision encoder
        if self.config.train_vision:
            for param in self.model.vision.parameters():
                param.requires_grad = True
            print("✓ Vision encoder: trainable")
        
        # Unfreeze projector
        if self.config.train_projector:
            for param in self.model.projector.parameters():
                param.requires_grad = True
            print("✓ Projector: trainable")
        
        # Keep language model frozen (or unfreeze if needed)
        if self.config.train_language:
            for param in self.model.language.parameters():
                param.requires_grad = True
            print("✓ Language model: trainable")
        else:
            print("✗ Language model: frozen")
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def prepare_data(self):
        """Prepare datasets for training."""
        from datasets import load_dataset, concatenate_datasets
        
        all_datasets = []
        for dataset_name in self.config.datasets:
            try:
                ds = load_dataset(dataset_name, split="train", streaming=True)
                all_datasets.append(ds)
                print(f"Loaded: {dataset_name}")
            except Exception as e:
                print(f"Warning: Could not load {dataset_name}: {e}")
        
        return all_datasets
    
    def train(self):
        """Run training loop."""
        if not HF_AVAILABLE:
            print("Error: transformers not installed")
            return
        
        print("=" * 60)
        print("Stage 1.5: ViT Incremental Learning")
        print("=" * 60)
        
        # Load and configure model
        self.load_model()
        self.configure_trainable_params()
        
        # Prepare data
        datasets = self.prepare_data()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_steps=self.config.max_steps,
            warmup_steps=self.config.warmup_steps,
            logging_steps=10,
            save_steps=500,
            bf16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=4,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
        )
        
        print(f"Starting training for {self.config.max_steps} steps...")
        trainer.train()
        
        # Save final model
        trainer.save_model(f"{self.config.output_dir}/final")
        print(f"Saved to {self.config.output_dir}/final")


def main():
    parser = argparse.ArgumentParser(description="Stage 1.5: ViT Incremental Learning")
    parser.add_argument("--variant", type=str, default="pixel")
    parser.add_argument("--model-path", type=str, default="./outputs/stage1/checkpoints/latest")
    parser.add_argument("--output-dir", type=str, default="./outputs/stage1.5")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    config = Stage1_5Config(
        variant=args.variant,
        model_path=args.model_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    
    trainer = ViTUnfreezeTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
