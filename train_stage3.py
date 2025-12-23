#!/usr/bin/env python3
"""
PixelLM Stage 3: Supervised Fine-Tuning (SFT).

Full parameter training on instruction-following data.
Goal: Conversation ability and task following.

Usage:
    python train_stage3.py --variant megapixel --checkpoint ./checkpoints/stage2
"""

import argparse
import logging

import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer

from src.model import PixelLMForCausalLM, get_variant_config, VARIANT_CONFIGS
from src.data import VLDataCollator, VLProcessor, DummyDataset, SFTDataset, create_sft_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Stage 3 datasets
SFT_DATASETS = [
    "liuhaotian/LLaVA-Instruct-150K",
    "Lin-Chen/ShareGPT4V",
]


def parse_args():
    parser = argparse.ArgumentParser(description="PixelLM Stage 3: SFT")
    parser.add_argument("--variant", type=str, default="pixel", choices=list(VARIANT_CONFIGS.keys()))
    parser.add_argument("--checkpoint", type=str, default=None, help="Stage 2 checkpoint")
    parser.add_argument("--datasets", type=str, nargs="+", default=SFT_DATASETS)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage3")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--use_dummy", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Config
    cfg = get_variant_config(args.variant)
    logger.info(f"Stage 3: SFT (Full Training) | Variant: {args.variant}")
    
    # Model
    logger.info("Creating model...")
    model = PixelLMForCausalLM.from_config(cfg)
    
    # Load Stage 2 checkpoint
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(f"{args.checkpoint}/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    # Full parameter training - all params trainable
    for param in model.parameters():
        param.requires_grad = True
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total: {total:,} | Trainable: {trainable:,} (100%)")
    
    # Dataset
    if args.use_dummy:
        train_dataset = DummyDataset(size=5000, seq_len=args.max_seq_len, image_size=cfg.vision.image_size)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except:
            tokenizer = None
        
        processor = VLProcessor(image_size=cfg.vision.image_size)
        train_dataset = create_sft_dataset(
            datasets=args.datasets,
            processor=processor,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
        )
    
    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=500,
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
    
    logger.info("Starting Stage 3: SFT (Full Training)...")
    trainer.train()
    trainer.save_model()
    logger.info("Stage 3 complete!")


if __name__ == "__main__":
    main()
