#!/usr/bin/env python3
"""
PixelLM Stage 1: Vision-Language Alignment.

Train projector only, freeze vision encoder and LLM.
Goal: Connect vision features to language embedding space.

Usage:
    python train_stage1.py --variant pixel --datasets conceptual_captions
"""

import argparse
import logging

import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer

from src.model import PixelLMForCausalLM, get_variant_config, VARIANT_CONFIGS
from src.data import VLDataCollator, VLProcessor, create_streaming_dataset, DummyDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Stage 1 datasets
ALIGNMENT_DATASETS = [
    "google-research-datasets/conceptual_captions",
]


def parse_args():
    parser = argparse.ArgumentParser(description="PixelLM Stage 1: Alignment")
    parser.add_argument("--variant", type=str, default="pixel", choices=list(VARIANT_CONFIGS.keys()))
    parser.add_argument("--datasets", type=str, nargs="+", default=ALIGNMENT_DATASETS)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage1")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--use_dummy", action="store_true")
    return parser.parse_args()


def freeze_except_projector(model):
    """Freeze everything except the projector."""
    # Freeze vision encoder
    for param in model.vision.parameters():
        param.requires_grad = False
    
    # Freeze language model
    for param in model.lm.parameters():
        param.requires_grad = False
    for param in model.lm_head.parameters():
        param.requires_grad = False
    
    # Keep projector trainable
    for param in model.projector.parameters():
        param.requires_grad = True
    
    # Keep image_newline trainable
    model.image_newline.requires_grad = True
    
    return model


def main():
    args = parse_args()
    
    # Config
    cfg = get_variant_config(args.variant)
    logger.info(f"Stage 1: Alignment | Variant: {args.variant}")
    
    # Model
    logger.info("Creating model...")
    model = PixelLMForCausalLM.from_config(cfg)
    model = freeze_except_projector(model)
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    
    # Dataset
    if args.use_dummy:
        train_dataset = DummyDataset(size=1000, seq_len=args.max_seq_len, image_size=cfg.vision.image_size)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except:
            tokenizer = None
        
        processor = VLProcessor(image_size=cfg.vision.image_size)
        train_dataset = create_streaming_dataset(
            stage="stage1_alignment",
            processor=processor,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            custom_datasets=args.datasets if args.datasets != ALIGNMENT_DATASETS else None,
        )
    
    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,  # No weight decay for alignment
        max_grad_norm=1.0,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=1000,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=VLDataCollator(),
    )
    
    logger.info("Starting Stage 1: Alignment...")
    trainer.train()
    trainer.save_model()
    logger.info("Stage 1 complete!")


if __name__ == "__main__":
    main()
