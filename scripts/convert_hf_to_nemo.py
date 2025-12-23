"""
Convert HuggingFace model to NeMo format.

Usage:
    python scripts/convert_hf_to_nemo.py \
        --input-path ./outputs/hf_model \
        --output-path ./outputs/nemo_model.nemo \
        --variant pixel
"""

import argparse
import os
from pathlib import Path

import torch


def convert_hf_to_nemo(
    input_path: str,
    output_path: str,
    variant: str = "pixel",
):
    """
    Convert HuggingFace PixelLM checkpoint to NeMo format.
    
    Args:
        input_path: Path to HuggingFace model directory
        output_path: Path to save NeMo checkpoint (.nemo)
        variant: Model variant (pixel, megapixel, gigapixel)
    """
    
    try:
        from nemo.collections.vlm import NevaModel
        from nemo.utils import AppState
        NEMO_AVAILABLE = True
    except ImportError:
        NEMO_AVAILABLE = False
        print("Error: NeMo not installed.")
        print("Install with: pip install nemo_toolkit[all]")
        return
    
    print(f"Converting {input_path} to NeMo format...")
    
    # Load HuggingFace model
    from src.model import PixelLM, get_config
    
    config = get_config(variant)
    hf_model = PixelLM(config)
    
    # Load weights
    state_dict_path = Path(input_path)
    if (state_dict_path / "pytorch_model.bin").exists():
        state_dict = torch.load(state_dict_path / "pytorch_model.bin", map_location="cpu")
    elif (state_dict_path / "model.safetensors").exists():
        from safetensors.torch import load_file
        state_dict = load_file(state_dict_path / "model.safetensors")
    else:
        # Try loading as single file
        state_dict = torch.load(input_path, map_location="cpu")
    
    hf_model.load_state_dict(state_dict, strict=False)
    
    # Map HuggingFace keys to NeMo keys
    nemo_state_dict = {}
    key_mapping = {
        # Vision encoder
        "vision.patch_embed.proj.weight": "vision_encoder.patch_embed.proj.weight",
        "vision.patch_embed.proj.bias": "vision_encoder.patch_embed.proj.bias",
        "vision.pos_embed": "vision_encoder.pos_embed",
        
        # Language model embeddings
        "language.embed.weight": "language_model.embedding.word_embeddings.weight",
        
        # LM head
        "lm_head.weight": "language_model.output_layer.weight",
    }
    
    for hf_key, param in hf_model.state_dict().items():
        # Apply key mapping
        nemo_key = key_mapping.get(hf_key)
        
        if nemo_key is None:
            # Default mapping: replace 'language.layers.' with 'language_model.decoder.layers.'
            nemo_key = hf_key.replace(
                "language.layers.", "language_model.decoder.layers."
            ).replace(
                ".attn.", ".self_attention."
            ).replace(
                ".ffn.", ".mlp."
            )
        
        nemo_state_dict[nemo_key] = param
    
    # Save NeMo checkpoint
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "state_dict": nemo_state_dict,
        "config": {
            "variant": variant,
            "hidden_size": config.language.hidden_size,
            "num_layers": config.language.num_layers,
            "num_heads": config.language.num_heads,
        },
    }, output_path)
    
    print(f"Saved NeMo checkpoint to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace to NeMo")
    parser.add_argument("--input-path", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--output-path", type=str, required=True, help="NeMo output path (.nemo)")
    parser.add_argument("--variant", type=str, default="pixel", choices=["pixel", "megapixel", "gigapixel"])
    args = parser.parse_args()
    
    convert_hf_to_nemo(args.input_path, args.output_path, args.variant)


if __name__ == "__main__":
    main()
