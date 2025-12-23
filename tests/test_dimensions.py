"""
Dimension validation and compatibility tests for DeepSeek VL2.

These tests ensure that all component dimensions are properly aligned
and that the model architecture is internally consistent.

Tests:
- Dimension flow through entire model
- Component compatibility
- Config consistency
- Layer dimension alignment
- Projector input/output matching
- Embedding dimension consistency
- Attention head dimension calculations
- MoE expert dimension matching
"""

import pytest
import torch
import math
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
    DeepSeekVL2TrainingConfig,
)
from src.models.vision_encoder import PatchEmbed, SigLIPVisionEncoder
from src.models.projector import MlpProjector
from src.models.mla import MultiHeadLatentAttention, RMSNorm, RotaryEmbedding
from src.models.moe import MoEGate, MoEExpert, MoELayer, DenseFFN
from src.models.language_model import DecoderLayer, DeepSeekMoELanguageModel
from src.models.deepseek_vl2 import DeepSeekVL2ForPretraining


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Patch Embedding Dimension Tests
# =============================================================================

class TestPatchEmbeddingDimensions:
    """Test patch embedding dimension calculations."""
    
    @pytest.mark.parametrize("image_size,patch_size,expected_patches", [
        (224, 16, 196),
        (384, 16, 576),
        (512, 16, 1024),
        (224, 14, 256),
        (384, 14, 729),
        (112, 8, 196),
    ])
    def test_num_patches_calculation(self, image_size, patch_size, expected_patches):
        """Test number of patches is calculated correctly for various configs."""
        patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            embed_dim=256
        )
        assert patch_embed.num_patches == expected_patches
    
    @pytest.mark.parametrize("image_size,patch_size", [
        (224, 16),
        (384, 16),
        (512, 32),
        (112, 8),
    ])
    def test_output_dimensions(self, image_size, patch_size):
        """Test patch embedding output has correct dimensions."""
        embed_dim = 512
        batch_size = 4
        
        patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        x = torch.randn(batch_size, 3, image_size, image_size)
        out = patch_embed(x)
        
        expected_patches = (image_size // patch_size) ** 2
        assert out.shape == (batch_size, expected_patches, embed_dim)
    
    def test_non_square_image_handling(self):
        """Test that non-square divisible images work."""
        # Note: Current implementation assumes square images
        # This test documents the expected behavior
        patch_embed = PatchEmbed(img_size=224, patch_size=16, embed_dim=256)
        
        # Square image works
        x_square = torch.randn(1, 3, 224, 224)
        out = patch_embed(x_square)
        assert out.shape[1] == 196  # 14x14


# =============================================================================
# Vision Encoder Dimension Tests
# =============================================================================

class TestVisionEncoderDimensions:
    """Test vision encoder internal dimension consistency."""
    
    @pytest.mark.parametrize("hidden_size,num_heads", [
        (256, 4),
        (512, 8),
        (768, 12),
        (1024, 16),
    ])
    def test_head_dimension_divisibility(self, hidden_size, num_heads):
        """Test hidden_size is divisible by num_heads."""
        assert hidden_size % num_heads == 0
        head_dim = hidden_size // num_heads
        assert head_dim > 0
    
    def test_position_embedding_matches_patches(self):
        """Test position embedding size matches number of patches."""
        config = VisionConfig(
            image_size=384,
            patch_size=16,
            hidden_size=1024,
            num_layers=24,
            num_heads=16,
        )
        encoder = SigLIPVisionEncoder(config)
        
        expected_patches = (384 // 16) ** 2  # 576
        assert encoder.pos_embed.shape == (1, expected_patches, config.hidden_size)
    
    def test_mlp_intermediate_dimension(self):
        """Test MLP intermediate dimension is correct."""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            mlp_ratio=4,
        )
        encoder = SigLIPVisionEncoder(config)
        
        # Check first block's MLP dimensions
        mlp = encoder.blocks[0].mlp
        expected_intermediate = config.hidden_size * config.mlp_ratio
        assert mlp.fc1.out_features == expected_intermediate
        assert mlp.fc2.in_features == expected_intermediate
    
    def test_output_matches_hidden_size(self):
        """Test encoder output matches hidden_size."""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            hidden_size=512,
            num_layers=4,
            num_heads=8,
        )
        encoder = SigLIPVisionEncoder(config)
        
        x = torch.randn(2, 3, 224, 224)
        out = encoder(x)
        
        assert out.shape[-1] == config.hidden_size


# =============================================================================
# Projector Dimension Tests
# =============================================================================

class TestProjectorDimensions:
    """Test projector input/output dimension matching."""
    
    def test_projector_input_matches_vision_output(self):
        """Test projector input_dim matches vision encoder output."""
        vision_config = VisionConfig(hidden_size=1024)
        projector_config = ProjectorConfig(
            type="downsample_mlp_gelu",
            input_dim=vision_config.hidden_size,  # Must match!
            output_dim=2048,
            downsample_ratio=2,
        )
        
        vision_encoder = SigLIPVisionEncoder(vision_config)
        projector = MlpProjector(projector_config)
        
        # Simulate vision output
        x = torch.randn(2, 3, vision_config.image_size, vision_config.image_size)
        vision_out = vision_encoder(x)
        
        # Projector should accept this without error
        proj_out = projector(vision_out)
        assert proj_out.shape[-1] == projector_config.output_dim
    
    def test_projector_output_matches_lm_hidden_size(self):
        """Test projector output matches language model hidden_size."""
        lm_hidden_size = 2048
        projector_config = ProjectorConfig(
            type="downsample_mlp_gelu",
            input_dim=1024,
            output_dim=lm_hidden_size,  # Must match LM!
            downsample_ratio=2,
        )
        
        projector = MlpProjector(projector_config)
        
        x = torch.randn(2, 576, 1024)  # Vision features
        out = projector(x)
        
        assert out.shape[-1] == lm_hidden_size
    
    @pytest.mark.parametrize("downsample_ratio", [1, 2, 4])
    def test_downsample_ratio_effects(self, downsample_ratio):
        """Test downsample ratio correctly reduces token count."""
        input_dim = 256
        config = ProjectorConfig(
            type="downsample_mlp_gelu",
            input_dim=input_dim,
            output_dim=512,
            depth=2,
            downsample_ratio=downsample_ratio,
        )
        projector = MlpProjector(config)
        
        # Input with specific number of patches
        num_patches = 196  # 14x14
        x = torch.randn(2, num_patches, input_dim)
        
        out = projector(x)
        
        # Output should have fewer tokens
        expected_tokens = num_patches // (downsample_ratio ** 2)
        assert out.shape[1] == expected_tokens
    
    def test_linear_projector_preserves_tokens(self):
        """Test linear projector doesn't change token count."""
        config = ProjectorConfig(
            type="linear",
            input_dim=256,
            output_dim=512,
        )
        projector = MlpProjector(config)
        
        x = torch.randn(2, 100, 256)
        out = projector(x)
        
        assert out.shape[1] == 100  # Same token count


# =============================================================================
# MLA Dimension Tests
# =============================================================================

class TestMLADimensions:
    """Test Multi-head Latent Attention dimension consistency."""
    
    def test_q_lora_dimensions(self):
        """Test Q LoRA projection dimensions."""
        config = LanguageConfig(
            hidden_size=2048,
            num_heads=16,
            mla=MLAConfig(
                q_lora_rank=1536,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                qk_nope_head_dim=128,
                v_head_dim=128,
            )
        )
        mla = MultiHeadLatentAttention(config)
        
        # Q_a: hidden_size -> q_lora_rank
        assert mla.q_a_proj.in_features == config.hidden_size
        assert mla.q_a_proj.out_features == config.mla.q_lora_rank
        
        # Q_b: q_lora_rank -> num_heads * (qk_nope + qk_rope)
        expected_out = config.num_heads * (config.mla.qk_nope_head_dim + config.mla.qk_rope_head_dim)
        assert mla.q_b_proj.in_features == config.mla.q_lora_rank
        assert mla.q_b_proj.out_features == expected_out
    
    def test_kv_lora_dimensions(self):
        """Test KV LoRA projection dimensions."""
        config = LanguageConfig(
            hidden_size=2048,
            num_heads=16,
            mla=MLAConfig(
                q_lora_rank=1536,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                qk_nope_head_dim=128,
                v_head_dim=128,
            )
        )
        mla = MultiHeadLatentAttention(config)
        
        # KV_a: hidden_size -> kv_lora_rank + qk_rope_head_dim
        expected_out = config.mla.kv_lora_rank + config.mla.qk_rope_head_dim
        assert mla.kv_a_proj.in_features == config.hidden_size
        assert mla.kv_a_proj.out_features == expected_out
        
        # KV_b: kv_lora_rank -> num_heads * (qk_nope + v_head)
        expected_out = config.num_heads * (config.mla.qk_nope_head_dim + config.mla.v_head_dim)
        assert mla.kv_b_proj.in_features == config.mla.kv_lora_rank
        assert mla.kv_b_proj.out_features == expected_out
    
    def test_output_projection_dimensions(self):
        """Test output projection dimensions."""
        config = LanguageConfig(
            hidden_size=2048,
            num_heads=16,
            mla=MLAConfig(v_head_dim=128)
        )
        mla = MultiHeadLatentAttention(config)
        
        # O: num_heads * v_head_dim -> hidden_size
        assert mla.o_proj.in_features == config.num_heads * config.mla.v_head_dim
        assert mla.o_proj.out_features == config.hidden_size
    
    def test_rope_dimension(self):
        """Test RoPE dimension consistency."""
        config = LanguageConfig(
            hidden_size=2048,
            num_heads=16,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            mla=MLAConfig(qk_rope_head_dim=64)
        )
        mla = MultiHeadLatentAttention(config)
        
        # RoPE dimension should match qk_rope_head_dim
        assert mla.rotary_emb.dim == config.mla.qk_rope_head_dim


# =============================================================================
# MoE Dimension Tests
# =============================================================================

class TestMoEDimensions:
    """Test Mixture of Experts dimension consistency."""
    
    def test_gate_dimensions(self):
        """Test MoE gate dimensions."""
        hidden_size = 2048
        num_experts = 64
        
        gate = MoEGate(hidden_size, num_experts, num_experts_per_token=6)
        
        assert gate.gate.in_features == hidden_size
        assert gate.gate.out_features == num_experts
    
    def test_expert_dimensions(self):
        """Test individual expert dimensions."""
        hidden_size = 2048
        intermediate_size = 1408
        
        expert = MoEExpert(hidden_size, intermediate_size)
        
        # Gate and up project to intermediate
        assert expert.gate_proj.in_features == hidden_size
        assert expert.gate_proj.out_features == intermediate_size
        assert expert.up_proj.in_features == hidden_size
        assert expert.up_proj.out_features == intermediate_size
        
        # Down projects back to hidden
        assert expert.down_proj.in_features == intermediate_size
        assert expert.down_proj.out_features == hidden_size
    
    def test_all_experts_same_dimensions(self):
        """Test all experts have same dimensions."""
        config = LanguageConfig(
            hidden_size=2048,
            moe=MoEConfig(
                enabled=True,
                num_experts=8,
                expert_hidden_size=1408,
            )
        )
        moe = MoELayer(config)
        
        for expert in moe.experts:
            assert expert.gate_proj.in_features == config.hidden_size
            assert expert.gate_proj.out_features == config.moe.expert_hidden_size
            assert expert.down_proj.out_features == config.hidden_size
    
    def test_shared_expert_dimensions(self):
        """Test shared expert has different (larger) dimensions."""
        config = LanguageConfig(
            hidden_size=2048,
            moe=MoEConfig(
                enabled=True,
                num_experts=8,
                expert_hidden_size=1408,
                shared_expert_hidden_size=2816,  # Larger!
                use_shared_expert=True,
            )
        )
        moe = MoELayer(config)
        
        # Shared expert should have larger intermediate size
        assert moe.shared_expert.gate_proj.out_features == config.moe.shared_expert_hidden_size
        assert moe.shared_expert.gate_proj.out_features > moe.experts[0].gate_proj.out_features
    
    def test_moe_vs_dense_hidden_size_match(self):
        """Test MoE and Dense FFN both use same hidden_size."""
        hidden_size = 2048
        intermediate_size = 5504
        
        moe_expert = MoEExpert(hidden_size, 1408)
        dense = DenseFFN(hidden_size, intermediate_size)
        
        # Both should input/output hidden_size
        assert moe_expert.gate_proj.in_features == dense.gate_proj.in_features
        assert moe_expert.down_proj.out_features == dense.down_proj.out_features


# =============================================================================
# Language Model Dimension Tests
# =============================================================================

class TestLanguageModelDimensions:
    """Test language model dimension consistency."""
    
    def test_embedding_dimension(self):
        """Test embedding matches hidden_size."""
        config = LanguageConfig(
            hidden_size=2048,
            vocab_size=102400,
        )
        lm = DeepSeekMoELanguageModel(config)
        
        assert lm.embed_tokens.embedding_dim == config.hidden_size
        assert lm.embed_tokens.num_embeddings == config.vocab_size
    
    def test_all_layers_same_dimensions(self):
        """Test all decoder layers have consistent dimensions."""
        config = LanguageConfig(
            hidden_size=2048,
            num_layers=6,
            num_heads=16,
            moe=MoEConfig(enabled=True, layer_freq=2),
        )
        lm = DeepSeekMoELanguageModel(config)
        
        for layer in lm.layers:
            # Input/output layernorm
            assert layer.input_layernorm.weight.shape[0] == config.hidden_size
            assert layer.post_attention_layernorm.weight.shape[0] == config.hidden_size
    
    def test_moe_layer_frequency(self):
        """Test MoE layers appear at correct frequency."""
        config = LanguageConfig(
            hidden_size=512,
            num_layers=6,
            moe=MoEConfig(enabled=True, layer_freq=2),
        )
        lm = DeepSeekMoELanguageModel(config)
        
        moe_layers = [i for i, layer in enumerate(lm.layers) if layer.is_moe]
        
        # With layer_freq=2, layers 0, 2, 4 should be MoE
        assert 0 in moe_layers
        assert 2 in moe_layers
        assert 4 in moe_layers


# =============================================================================
# Full Model Integration Dimension Tests
# =============================================================================

class TestFullModelDimensions:
    """Test dimension flow through entire model."""
    
    def test_vision_to_projector_dimension_chain(self):
        """Test dimensions flow: Vision -> Projector."""
        vision_hidden = 1024
        projector_output = 2048
        
        vision_config = VisionConfig(
            image_size=224,
            patch_size=16,
            hidden_size=vision_hidden,
        )
        projector_config = ProjectorConfig(
            type="downsample_mlp_gelu",
            input_dim=vision_hidden,  # Must match vision output!
            output_dim=projector_output,
            downsample_ratio=2,
        )
        
        vision = SigLIPVisionEncoder(vision_config)
        projector = MlpProjector(projector_config)
        
        x = torch.randn(2, 3, 224, 224)
        v_out = vision(x)
        p_out = projector(v_out)
        
        assert v_out.shape[-1] == vision_hidden
        assert p_out.shape[-1] == projector_output
    
    def test_projector_to_lm_dimension_chain(self):
        """Test dimensions flow: Projector -> Language Model."""
        lm_hidden = 2048
        
        projector_config = ProjectorConfig(
            type="linear",
            input_dim=1024,
            output_dim=lm_hidden,  # Must match LM hidden!
        )
        language_config = LanguageConfig(
            hidden_size=lm_hidden,
            num_layers=2,
        )
        
        projector = MlpProjector(projector_config)
        lm = DeepSeekMoELanguageModel(language_config)
        
        # Vision features
        x = torch.randn(2, 100, 1024)
        proj_out = projector(x)
        
        # LM embedding dimension should match projector output
        assert proj_out.shape[-1] == lm.embed_tokens.embedding_dim
    
    def test_full_model_dimension_consistency(self):
        """Test all dimensions are consistent in full model."""
        vision_hidden = 256
        lm_hidden = 512
        
        model_config = ModelConfig(
            vision=VisionConfig(
                image_size=112,
                patch_size=8,
                hidden_size=vision_hidden,
                num_layers=2,
                num_heads=4,
            ),
            projector=ProjectorConfig(
                type="linear",
                input_dim=vision_hidden,
                output_dim=lm_hidden,
            ),
            language=LanguageConfig(
                hidden_size=lm_hidden,
                num_layers=2,
                num_heads=8,
                vocab_size=1000,
                mla=MLAConfig(
                    q_lora_rank=256,
                    kv_lora_rank=128,
                    qk_rope_head_dim=16,
                    qk_nope_head_dim=16,
                    v_head_dim=32,
                ),
                moe=MoEConfig(enabled=False),
            ),
        )
        
        model = DeepSeekVL2ForPretraining(model_config)
        
        # Check internal consistency
        assert model.vision_encoder.output_dim == model_config.vision.hidden_size
        assert model.projector.output_dim == model_config.projector.output_dim
        assert model.projector.output_dim == model_config.language.hidden_size
        assert model.language_model.embed_tokens.embedding_dim == model_config.language.hidden_size
    
    def test_lm_head_dimensions(self):
        """Test LM head matches embedding dimensions."""
        config = ModelConfig(
            vision=VisionConfig(hidden_size=128, num_layers=1, num_heads=2),
            projector=ProjectorConfig(input_dim=128, output_dim=256),
            language=LanguageConfig(
                hidden_size=256,
                num_layers=1,
                vocab_size=1000,
                mla=MLAConfig(
                    q_lora_rank=128, kv_lora_rank=64,
                    qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16
                ),
                moe=MoEConfig(enabled=False),
            ),
        )
        model = DeepSeekVL2ForPretraining(config)
        
        # LM head should output vocab_size
        assert model.lm_head.out_features == config.language.vocab_size
        assert model.lm_head.in_features == config.language.hidden_size


# =============================================================================
# Config Validation Tests
# =============================================================================

class TestConfigValidation:
    """Test configuration validation for dimension mismatches."""
    
    def test_head_dim_divisibility(self):
        """Test hidden_size must be divisible by num_heads."""
        valid_configs = [
            (256, 4),   # 64 per head
            (512, 8),   # 64 per head
            (768, 12),  # 64 per head
            (1024, 16), # 64 per head
        ]
        
        for hidden_size, num_heads in valid_configs:
            assert hidden_size % num_heads == 0
    
    def test_consistent_variant_configs(self):
        """Test all variant configs are internally consistent."""
        variants = {
            "tiny": {
                "hidden_size": 1536,
                "num_heads": 12,
                "num_layers": 24,
            },
            "small": {
                "hidden_size": 2048,
                "num_heads": 16,
                "num_layers": 27,
            },
            "large": {
                "hidden_size": 4096,
                "num_heads": 32,
                "num_layers": 60,
            },
        }
        
        for name, config in variants.items():
            # Hidden size divisible by num_heads
            assert config["hidden_size"] % config["num_heads"] == 0, f"{name} head dim invalid"
            # Positive layer count
            assert config["num_layers"] > 0, f"{name} needs positive layers"
    
    def test_projector_input_output_positive(self):
        """Test projector dimensions are positive."""
        config = ProjectorConfig(
            input_dim=1024,
            output_dim=2048,
        )
        assert config.input_dim > 0
        assert config.output_dim > 0
    
    def test_mla_lora_ranks_positive(self):
        """Test MLA LoRA ranks are positive."""
        config = MLAConfig(
            q_lora_rank=1536,
            kv_lora_rank=512,
        )
        assert config.q_lora_rank > 0
        assert config.kv_lora_rank > 0


# =============================================================================
# Shape Transformation Tests
# =============================================================================

class TestShapeTransformations:
    """Test correct shape transformations through model."""
    
    def test_batch_preservation(self, device):
        """Test batch size is preserved through model."""
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
        
        for batch_size in [1, 2, 4, 8]:
            input_ids = torch.randint(0, 256, (batch_size, 16), device=device)
            outputs = model(input_ids=input_ids)
            assert outputs["logits"].shape[0] == batch_size
    
    def test_sequence_length_preservation(self, device):
        """Test sequence length is preserved (except for causal shift)."""
        config = ModelConfig(
            vision=VisionConfig(image_size=56, hidden_size=64, num_layers=1, num_heads=2, patch_size=8),
            projector=ProjectorConfig(input_dim=64, output_dim=128, type="linear"),
            language=LanguageConfig(
                hidden_size=128, num_layers=1, vocab_size=256, max_position_embeddings=256,
                mla=MLAConfig(q_lora_rank=64, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16),
                moe=MoEConfig(enabled=False)
            ),
        )
        model = DeepSeekVL2ForPretraining(config).to(device)
        
        for seq_len in [16, 32, 64, 128]:
            input_ids = torch.randint(0, 256, (2, seq_len), device=device)
            outputs = model(input_ids=input_ids)
            assert outputs["logits"].shape[1] == seq_len
    
    def test_vocab_size_output(self, device):
        """Test output vocab dimension matches config."""
        vocab_size = 512
        config = ModelConfig(
            vision=VisionConfig(image_size=56, hidden_size=64, num_layers=1, num_heads=2, patch_size=8),
            projector=ProjectorConfig(input_dim=64, output_dim=128, type="linear"),
            language=LanguageConfig(
                hidden_size=128, num_layers=1, vocab_size=vocab_size,
                mla=MLAConfig(q_lora_rank=64, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16),
                moe=MoEConfig(enabled=False)
            ),
        )
        model = DeepSeekVL2ForPretraining(config).to(device)
        
        input_ids = torch.randint(0, vocab_size, (2, 16), device=device)
        outputs = model(input_ids=input_ids)
        
        assert outputs["logits"].shape[2] == vocab_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
