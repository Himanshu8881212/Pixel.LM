"""
Convert NeMo model to HuggingFace format.

Usage:
    python scripts/convert_nemo_to_hf.py \
        --input-path ./outputs/nemo_model.nemo \
        --output-path ./outputs/hf_model \
        --variant pixel
"""

import argparse
import os
from pathlib import Path
import json

import torch


def convert_nemo_to_hf(
    input_path: str,
    output_path: str,
    variant: str = "pixel",
):
    """
    Convert NeMo checkpoint to HuggingFace PixelLM format.
    
    Args:
        input_path: Path to NeMo checkpoint (.nemo)
        output_path: Path to save HuggingFace model directory
        variant: Model variant (pixel, megapixel, gigapixel)
    """
    
    print(f"Converting {input_path} to HuggingFace format...")
    
    # Load NeMo checkpoint
    checkpoint = torch.load(input_path, map_location="cpu")
    nemo_state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Map NeMo keys to HuggingFace keys
    hf_state_dict = {}
    key_mapping = {
        # Vision encoder
        "vision_encoder.patch_embed.proj.weight": "vision.patch_embed.proj.weight",
        "vision_encoder.patch_embed.proj.bias": "vision.patch_embed.proj.bias",
        "vision_encoder.pos_embed": "vision.pos_embed",
        
        # Language model embeddings
        "language_model.embedding.word_embeddings.weight": "language.embed.weight",
        
        # LM head
        "language_model.output_layer.weight": "lm_head.weight",
    }
    
    for nemo_key, param in nemo_state_dict.items():
        # Apply reverse key mapping
        hf_key = None
        for nemo_pattern, hf_pattern in key_mapping.items():
            if nemo_key == nemo_pattern:
                hf_key = hf_pattern
                break
        
        if hf_key is None:
            # Default mapping
            hf_key = nemo_key.replace(
                "language_model.decoder.layers.", "language.layers."
            ).replace(
                ".self_attention.", ".attn."
            ).replace(
                ".mlp.", ".ffn."
            )
        
        hf_state_dict[hf_key] = param
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    torch.save(hf_state_dict, output_dir / "pytorch_model.bin")
    
    # Create config.json
    from src.model import get_config
    config = get_config(variant)
    
    hf_config = {
        "model_type": "pixellm",
        "variant": variant,
        "hidden_size": config.language.hidden_size,
        "num_hidden_layers": config.language.num_layers,
        "num_attention_heads": config.language.num_heads,
        "intermediate_size": config.language.intermediate_size,
        "vocab_size": config.language.vocab_size,
        "max_position_embeddings": config.language.max_position_embeddings,
        "vision_config": {
            "image_size": config.vision.image_size,
            "patch_size": config.vision.patch_size,
            "hidden_size": config.vision.hidden_size,
            "num_layers": config.vision.num_layers,
        },
        "moe_config": {
            "num_experts": config.language.moe.num_experts,
            "num_experts_per_token": config.language.moe.num_experts_per_token,
        },
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)
    
    print(f"Saved HuggingFace model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert NeMo to HuggingFace")
    parser.add_argument("--input-path", type=str, required=True, help="NeMo checkpoint path (.nemo)")
    parser.add_argument("--output-path", type=str, required=True, help="HuggingFace output directory")
    parser.add_argument("--variant", type=str, default="pixel", choices=["pixel", "megapixel", "gigapixel"])
    args = parser.parse_args()
    
    convert_nemo_to_hf(args.input_path, args.output_path, args.variant)


if __name__ == "__main__":
    main()
