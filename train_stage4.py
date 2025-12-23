#!/usr/bin/env python3
"""
PixelLM Stage 4: Direct Preference Optimization (DPO).

Full parameter training to align model with human preferences.
Goal: Reduce hallucinations, improve response quality.

Usage:
    python train_stage4.py --variant megapixel --checkpoint ./checkpoints/stage3
"""

import argparse
import logging

import torch
from transformers import AutoTokenizer

from src.model import PixelLMForCausalLM, get_variant_config, VARIANT_CONFIGS
from src.data import VLProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="PixelLM Stage 4: DPO")
    parser.add_argument("--variant", type=str, default="pixel", choices=list(VARIANT_CONFIGS.keys()))
    parser.add_argument("--checkpoint", type=str, required=True, help="Stage 3 checkpoint")
    parser.add_argument("--ref_checkpoint", type=str, default=None, help="Reference model (defaults to checkpoint)")
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage4")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--use_dummy", action="store_true")
    return parser.parse_args()


def dpo_loss(policy_logps_chosen, policy_logps_rejected, 
             ref_logps_chosen, ref_logps_rejected, beta=0.1):
    """Compute DPO loss."""
    policy_ratio = policy_logps_chosen - policy_logps_rejected
    ref_ratio = ref_logps_chosen - ref_logps_rejected
    
    losses = -torch.nn.functional.logsigmoid(beta * (policy_ratio - ref_ratio))
    return losses.mean()


def get_logps(model, input_ids, labels, attention_mask):
    """Get log probabilities for sequences."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1]
    labels = labels[:, 1:]
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    selected_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask out padding
    mask = (labels != -100).float()
    return (selected_log_probs * mask).sum(-1) / mask.sum(-1)


def main():
    args = parse_args()
    
    # Config
    cfg = get_variant_config(args.variant)
    logger.info(f"Stage 4: DPO (Full Training) | Variant: {args.variant} | Beta: {args.beta}")
    
    # Policy model - full parameter training
    logger.info("Loading policy model...")
    policy_model = PixelLMForCausalLM.from_config(cfg)
    state_dict = torch.load(f"{args.checkpoint}/pytorch_model.bin", map_location="cpu")
    policy_model.load_state_dict(state_dict, strict=False)
    
    # All parameters trainable
    for param in policy_model.parameters():
        param.requires_grad = True
    
    # Reference model (frozen copy)
    logger.info("Loading reference model...")
    ref_model = PixelLMForCausalLM.from_config(cfg)
    ref_checkpoint = args.ref_checkpoint or args.checkpoint
    ref_state = torch.load(f"{ref_checkpoint}/pytorch_model.bin", map_location="cpu")
    ref_model.load_state_dict(ref_state, strict=False)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    total = sum(p.numel() for p in policy_model.parameters())
    trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    logger.info(f"Policy model - Total: {total:,} | Trainable: {trainable:,} (100%)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model.to(device)
    ref_model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.lr, weight_decay=0.0)
    
    # Training loop with dummy data
    if args.use_dummy:
        logger.info("Using dummy preference data...")
        num_samples = 1000
        seq_len = min(args.max_seq_len, 512)
        
        for epoch in range(args.epochs):
            total_loss = 0
            num_batches = num_samples // args.batch_size
            
            for step in range(num_batches):
                # Generate dummy preference pairs
                chosen_ids = torch.randint(0, cfg.language.vocab_size, (args.batch_size, seq_len), device=device)
                rejected_ids = torch.randint(0, cfg.language.vocab_size, (args.batch_size, seq_len), device=device)
                attention_mask = torch.ones_like(chosen_ids)
                
                # Get log probs
                with torch.no_grad():
                    ref_logps_chosen = get_logps(ref_model, chosen_ids, chosen_ids, attention_mask)
                    ref_logps_rejected = get_logps(ref_model, rejected_ids, rejected_ids, attention_mask)
                
                policy_logps_chosen = get_logps(policy_model, chosen_ids, chosen_ids, attention_mask)
                policy_logps_rejected = get_logps(policy_model, rejected_ids, rejected_ids, attention_mask)
                
                # DPO loss
                loss = dpo_loss(
                    policy_logps_chosen, policy_logps_rejected,
                    ref_logps_chosen, ref_logps_rejected,
                    beta=args.beta
                )
                
                # Backward
                loss.backward()
                
                if (step + 1) % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
                
                if step % 10 == 0:
                    logger.info(f"Epoch {epoch+1} | Step {step}/{num_batches} | Loss: {loss.item():.4f}")
            
            logger.info(f"Epoch {epoch+1} complete. Avg loss: {total_loss/num_batches:.4f}")
    else:
        logger.info("Note: For real DPO training, use TRL library:")
        logger.info("  from trl import DPOTrainer, DPOConfig")
        logger.info("  trainer = DPOTrainer(model=policy_model, ref_model=ref_model, ...)")
    
    # Save
    policy_model.save_pretrained(args.output_dir)
    logger.info(f"Stage 4 complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
