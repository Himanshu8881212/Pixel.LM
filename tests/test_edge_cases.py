"""
Edge cases and error handling tests for DeepSeek VL2.

Tests:
- Input boundary conditions
- Error handling
- Edge cases in dimensions
- Numerical edge cases
"""

import pytest
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    VisionConfig,
    ProjectorConfig,
    LanguageConfig,
    ModelConfig,
    MLAConfig,
    MoEConfig,
)
from src.models.vision_encoder import SigLIPVisionEncoder, PatchEmbed
from src.models.projector import MlpProjector
from src.models.mla import MultiHeadLatentAttention, RMSNorm
from src.models.moe import MoELayer
from src.models.language_model import DeepSeekMoELanguageModel
from src.models.deepseek_vl2 import DeepSeekVL2ForPretraining


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Input Boundary Tests
# =============================================================================

class TestInputBoundaries:
    """Test input boundary conditions."""
    
    def test_batch_size_one(self, device):
        """Test batch size of 1 works."""
        config = ModelConfig(
            vision=VisionConfig(image_size=56, hidden_size=64, num_layers=1, num_heads=2, patch_size=8),
            projector=ProjectorConfig(input_dim=64, output_dim=128, type="linear"),
            language=LanguageConfig(
                hidden_size=128, num_layers=1, vocab_size=256,
                mla=MLAConfig(q_lora_rank=64, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16),
                moe=MoEConfig(enabled=False)
            ),
        )
        model = DeepSeekVL2ForPretraining(config).to(device)
        
        input_ids = torch.randint(0, 256, (1, 16), device=device)
        outputs = model(input_ids=input_ids)
        
        assert outputs["logits"].shape[0] == 1
    
    def test_sequence_length_one(self, device):
        """Test sequence length of 1 works."""
        config = ModelConfig(
            vision=VisionConfig(image_size=56, hidden_size=64, num_layers=1, num_heads=2, patch_size=8),
            projector=ProjectorConfig(input_dim=64, output_dim=128, type="linear"),
            language=LanguageConfig(
                hidden_size=128, num_layers=1, vocab_size=256,
                mla=MLAConfig(q_lora_rank=64, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16),
                moe=MoEConfig(enabled=False)
            ),
        )
        model = DeepSeekVL2ForPretraining(config).to(device)
        
        input_ids = torch.randint(0, 256, (2, 1), device=device)
        outputs = model(input_ids=input_ids)
        
        assert outputs["logits"].shape[1] == 1
    
    def test_single_image_patch(self, device):
        """Test minimal image that produces single patch."""
        # 8x8 image with 8x8 patch = 1 patch
        config = VisionConfig(
            image_size=8,
            patch_size=8,
            hidden_size=64,
            num_layers=1,
            num_heads=2,
        )
        encoder = SigLIPVisionEncoder(config).to(device)
        
        x = torch.randn(1, 3, 8, 8, device=device)
        out = encoder(x)
        
        assert out.shape == (1, 1, 64)  # 1 patch
    
    def test_max_experts_selected(self, device):
        """Test when selecting all experts."""
        num_experts = 4
        config = LanguageConfig(
            hidden_size=256,
            moe=MoEConfig(
                enabled=True,
                num_experts=num_experts,
                num_experts_per_token=num_experts,  # Select all!
            )
        )
        moe = MoELayer(config).to(device)
        
        x = torch.randn(2, 8, 256, device=device)
        out = moe(x)
        
        assert out.shape == x.shape
    
    def test_single_expert(self, device):
        """Test with single expert."""
        config = LanguageConfig(
            hidden_size=256,
            moe=MoEConfig(
                enabled=True,
                num_experts=1,
                num_experts_per_token=1,
            )
        )
        moe = MoELayer(config).to(device)
        
        x = torch.randn(2, 8, 256, device=device)
        out = moe(x)
        
        assert out.shape == x.shape


# =============================================================================
# Numerical Edge Cases
# =============================================================================

class TestNumericalEdgeCases:
    """Test numerical edge cases."""
    
    def test_zero_input(self, device):
        """Test model handles zero input."""
        config = ModelConfig(
            vision=VisionConfig(image_size=56, hidden_size=64, num_layers=1, num_heads=2, patch_size=8),
            projector=ProjectorConfig(input_dim=64, output_dim=128, type="linear"),
            language=LanguageConfig(
                hidden_size=128, num_layers=1, vocab_size=256,
                mla=MLAConfig(q_lora_rank=64, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16),
                moe=MoEConfig(enabled=False)
            ),
        )
        model = DeepSeekVL2ForPretraining(config).to(device)
        
        # Zero token ID is valid
        input_ids = torch.zeros(2, 16, dtype=torch.long, device=device)
        outputs = model(input_ids=input_ids)
        
        assert torch.isfinite(outputs["logits"]).all()
    
    def test_rmsnorm_zero_input(self, device):
        """Test RMSNorm handles zero input."""
        norm = RMSNorm(256, eps=1e-6).to(device)
        
        x = torch.zeros(2, 10, 256, device=device)
        out = norm(x)
        
        # Should be zero (or very close due to eps)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)
    
    def test_rmsnorm_large_values(self, device):
        """Test RMSNorm handles large values."""
        norm = RMSNorm(256).to(device)
        
        x = torch.randn(2, 10, 256, device=device) * 1000
        out = norm(x)
        
        # Should be normalized to reasonable range
        assert out.abs().mean() < 100
    
    def test_attention_with_zero_queries(self, device):
        """Test attention handles near-zero queries."""
        config = LanguageConfig(
            hidden_size=128, num_heads=4,
            mla=MLAConfig(q_lora_rank=64, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16),
        )
        mla = MultiHeadLatentAttention(config).to(device)
        
        # Very small input (near zero)
        x = torch.randn(2, 16, 128, device=device) * 1e-8
        out, _ = mla(x)
        
        assert torch.isfinite(out).all()


# =============================================================================
# Projector Edge Cases
# =============================================================================

class TestProjectorEdgeCases:
    """Test projector edge cases."""
    
    def test_non_square_patches(self):
        """Test projector with non-square-number of patches."""
        config = ProjectorConfig(
            type="downsample_mlp_gelu",
            input_dim=256,
            output_dim=512,
            downsample_ratio=2,
        )
        projector = MlpProjector(config)
        
        # 225 patches (15x15) - needs padding for 2x downsample
        x = torch.randn(2, 225, 256)
        out = projector(x)
        
        # Should handle padding
        assert out.shape[0] == 2
        assert out.shape[2] == 512
    
    def test_downsample_ratio_larger_than_image(self):
        """Test downsample with ratio larger than patch grid."""
        config = ProjectorConfig(
            type="downsample_mlp_gelu",
            input_dim=256,
            output_dim=512,
            downsample_ratio=4,
        )
        projector = MlpProjector(config)
        
        # 4 patches (2x2) with ratio 4 should give 1 output
        # But 2x2 < 4x4, so needs heavy padding
        x = torch.randn(2, 4, 256)
        out = projector(x)
        
        # Should produce at least one output
        assert out.shape[0] == 2


# =============================================================================
# Label Edge Cases
# =============================================================================

class TestLabelEdgeCases:
    """Test label handling edge cases."""
    
    def test_all_ignore_labels(self, device):
        """Test when all labels are -100 (ignore)."""
        config = ModelConfig(
            vision=VisionConfig(image_size=56, hidden_size=64, num_layers=1, num_heads=2, patch_size=8),
            projector=ProjectorConfig(input_dim=64, output_dim=128, type="linear"),
            language=LanguageConfig(
                hidden_size=128, num_layers=1, vocab_size=256,
                mla=MLAConfig(q_lora_rank=64, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16),
                moe=MoEConfig(enabled=False)
            ),
        )
        model = DeepSeekVL2ForPretraining(config).to(device)
        
        input_ids = torch.randint(0, 256, (2, 16), device=device)
        labels = torch.full((2, 16), -100, device=device)
        
        outputs = model(input_ids=input_ids, labels=labels)
        
        # NaN or 0 loss is acceptable when all labels ignored
        assert outputs["loss"] is not None
    
    def test_single_valid_label(self, device):
        """Test with only one valid label."""
        config = ModelConfig(
            vision=VisionConfig(image_size=56, hidden_size=64, num_layers=1, num_heads=2, patch_size=8),
            projector=ProjectorConfig(input_dim=64, output_dim=128, type="linear"),
            language=LanguageConfig(
                hidden_size=128, num_layers=1, vocab_size=256,
                mla=MLAConfig(q_lora_rank=64, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16),
                moe=MoEConfig(enabled=False)
            ),
        )
        model = DeepSeekVL2ForPretraining(config).to(device)
        
        input_ids = torch.randint(0, 256, (2, 16), device=device)
        labels = torch.full((2, 16), -100, device=device)
        labels[:, -1] = torch.randint(0, 256, (2,), device=device)  # Only last token
        
        outputs = model(input_ids=input_ids, labels=labels)
        
        assert torch.isfinite(outputs["loss"])


# =============================================================================
# Image Position Edge Cases
# =============================================================================

class TestImagePositionEdgeCases:
    """Test image position handling edge cases."""
    
    def test_image_at_start(self, device):
        """Test image placed at start of sequence."""
        config = ModelConfig(
            vision=VisionConfig(image_size=56, hidden_size=64, num_layers=1, num_heads=2, patch_size=8),
            projector=ProjectorConfig(input_dim=64, output_dim=128, type="linear"),
            language=LanguageConfig(
                hidden_size=128, num_layers=1, vocab_size=256,
                mla=MLAConfig(q_lora_rank=64, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16),
                moe=MoEConfig(enabled=False)
            ),
        )
        model = DeepSeekVL2ForPretraining(config).to(device)
        
        input_ids = torch.randint(0, 256, (1, 64), device=device)
        images = torch.randn(1, 3, 56, 56, device=device)
        image_positions = torch.tensor([[0]], device=device)  # Start
        
        outputs = model(
            input_ids=input_ids,
            images=images,
            image_positions=image_positions,
        )
        
        assert outputs["logits"].shape[0] == 1
    
    def test_image_at_end(self, device):
        """Test image placed at end of sequence."""
        config = ModelConfig(
            vision=VisionConfig(image_size=56, hidden_size=64, num_layers=1, num_heads=2, patch_size=8),
            projector=ProjectorConfig(input_dim=64, output_dim=128, type="linear"),
            language=LanguageConfig(
                hidden_size=128, num_layers=1, vocab_size=256,
                mla=MLAConfig(q_lora_rank=64, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16),
                moe=MoEConfig(enabled=False)
            ),
        )
        model = DeepSeekVL2ForPretraining(config).to(device)
        
        seq_len = 64
        input_ids = torch.randint(0, 256, (1, seq_len), device=device)
        images = torch.randn(1, 3, 56, 56, device=device)
        # Position near end (but leaving room for image tokens)
        image_positions = torch.tensor([[seq_len - 10]], device=device)
        
        outputs = model(
            input_ids=input_ids,
            images=images,
            image_positions=image_positions,
        )
        
        assert outputs["logits"].shape[0] == 1
    
    def test_negative_image_position(self, device):
        """Test negative image position (should be skipped)."""
        config = ModelConfig(
            vision=VisionConfig(image_size=56, hidden_size=64, num_layers=1, num_heads=2, patch_size=8),
            projector=ProjectorConfig(input_dim=64, output_dim=128, type="linear"),
            language=LanguageConfig(
                hidden_size=128, num_layers=1, vocab_size=256,
                mla=MLAConfig(q_lora_rank=64, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16),
                moe=MoEConfig(enabled=False)
            ),
        )
        model = DeepSeekVL2ForPretraining(config).to(device)
        
        input_ids = torch.randint(0, 256, (1, 32), device=device)
        images = torch.randn(1, 3, 56, 56, device=device)
        image_positions = torch.tensor([[-1]], device=device)  # Negative = skip
        
        outputs = model(
            input_ids=input_ids,
            images=images,
            image_positions=image_positions,
        )
        
        assert outputs["logits"].shape[0] == 1


# =============================================================================
# Memory Edge Cases
# =============================================================================

class TestMemoryEdgeCases:
    """Test memory handling edge cases."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_large_batch_gradient_checkpointing(self, device):
        """Test gradient checkpointing with larger batch."""
        config = ModelConfig(
            vision=VisionConfig(image_size=56, hidden_size=64, num_layers=2, num_heads=2, patch_size=8),
            projector=ProjectorConfig(input_dim=64, output_dim=128, type="linear"),
            language=LanguageConfig(
                hidden_size=128, num_layers=2, vocab_size=256,
                mla=MLAConfig(q_lora_rank=64, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16),
                moe=MoEConfig(enabled=False)
            ),
        )
        model = DeepSeekVL2ForPretraining(config).to(device)
        model.enable_gradient_checkpointing()
        
        # Larger batch
        input_ids = torch.randint(0, 256, (8, 64), device=device)
        labels = input_ids.clone()
        
        outputs = model(input_ids=input_ids, labels=labels)
        outputs["loss"].backward()
        
        # Should complete without OOM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
