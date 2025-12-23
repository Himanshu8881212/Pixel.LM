#!/usr/bin/env python3
"""
PixelLM Training.

Variants:
    - pixel: ~1B params (dense, research/testing)
    - megapixel: ~7B params (8 experts MoE, production)
    - gigapixel: ~27B params (64 experts MoE, SOTA)
    - tiny: for unit tests

Usage:
    python train.py --variant pixel
    python train.py --variant megapixel --datasets laion/laion400m kakaobrain/coyo-700m
    accelerate launch train.py --variant gigapixel
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer

from src.model import PixelLMForCausalLM, get_variant_config, VARIANT_CONFIGS
from src.data import VLDataCollator, DummyDataset, VLProcessor, create_streaming_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="PixelLM Training")
    parser.add_argument("--variant", type=str, default="tiny", choices=list(VARIANT_CONFIGS.keys()),
                        help="Model variant: pixel, megapixel, gigapixel, tiny")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="HuggingFace dataset names for streaming")
    parser.add_argument("--stage", type=str, default="full",
                        choices=["stage1_alignment", "stage2_pretrain", "stage3_finetune", "full"],
                        help="Pretraining stage (determines default datasets)")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        help="Tokenizer name or path")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--use_dummy", action="store_true",
                        help="Use dummy dataset instead of streaming")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 for epochs-based)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Get model config
    cfg = get_variant_config(args.variant)
    logger.info(f"Using variant: {args.variant}")
    
    # Create model
    logger.info("Creating model...")
    model = PixelLMForCausalLM.from_config(cfg)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total:,}")
    logger.info(f"Trainable parameters: {trainable:,}")
    
    # Dataset
    logger.info("Creating dataset...")
    if args.use_dummy:
        train_dataset = DummyDataset(
            size=1000,
            seq_len=min(args.max_seq_len, 512),
            image_size=cfg.vision.image_size,
            vocab_size=cfg.language.vocab_size
        )
    else:
        # Load tokenizer for streaming
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}. Using dummy tokens.")
            tokenizer = None
        
        processor = VLProcessor(image_size=cfg.vision.image_size)
        train_dataset = create_streaming_dataset(
            stage=args.stage,
            processor=processor,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            custom_datasets=args.datasets,
        )
        logger.info(f"Streaming from: {args.datasets or args.stage}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
        gradient_checkpointing=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=VLDataCollator(),
    )
    
    logger.info("Starting training...")
    trainer.train()
    trainer.save_model()
    logger.info("Done!")


if __name__ == "__main__":
    main()
