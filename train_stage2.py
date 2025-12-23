#!/usr/bin/env python3
"""
PixelLM Stage 2: Vision-Language Pretraining.

Train all components on large-scale image-text data.
Goal: Deep multimodal understanding.

Usage:
    python train_stage2.py --variant megapixel --max_steps 100000
    accelerate launch train_stage2.py --variant gigapixel
"""

import argparse
import logging

import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer

from src.model import PixelLMForCausalLM, get_variant_config, VARIANT_CONFIGS
from src.data import VLDataCollator, VLProcessor, create_streaming_dataset, DummyDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Stage 2 datasets (large-scale)
PRETRAIN_DATASETS = [
    "laion/laion400m",
    "kakaobrain/coyo-700m",
]


def parse_args():
    parser = argparse.ArgumentParser(description="PixelLM Stage 2: Pretraining")
    parser.add_argument("--variant", type=str, default="pixel", choices=list(VARIANT_CONFIGS.keys()))
    parser.add_argument("--checkpoint", type=str, default=None, help="Stage 1 checkpoint to resume from")
    parser.add_argument("--datasets", type=str, nargs="+", default=PRETRAIN_DATASETS)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage2")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--use_dummy", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Config
    cfg = get_variant_config(args.variant)
    logger.info(f"Stage 2: Pretraining | Variant: {args.variant}")
    
    # Model
    logger.info("Creating model...")
    model = PixelLMForCausalLM.from_config(cfg)
    
    # Load Stage 1 checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(f"{args.checkpoint}/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    # All parameters trainable
    for param in model.parameters():
        param.requires_grad = True
    
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total:,}")
    
    # Dataset
    if args.use_dummy:
        train_dataset = DummyDataset(size=10000, seq_len=args.max_seq_len, image_size=cfg.vision.image_size)
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except:
            tokenizer = None
        
        processor = VLProcessor(image_size=cfg.vision.image_size)
        train_dataset = create_streaming_dataset(
            stage="stage2_pretrain",
            processor=processor,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            custom_datasets=args.datasets if args.datasets != PRETRAIN_DATASETS else None,
        )
    
    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.01,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=5000,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=4,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=VLDataCollator(),
    )
    
    logger.info("Starting Stage 2: Pretraining...")
    trainer.train()
    trainer.save_model()
    logger.info("Stage 2 complete!")


if __name__ == "__main__":
    main()
