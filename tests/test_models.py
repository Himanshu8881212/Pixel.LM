"""
Model architecture tests for DeepSeek VL2.

Tests:
- Vision encoder architecture and forward pass
- Multi-head Latent Attention (MLA)
- Mixture of Experts (MoE) routing and computation
- VL Projector
- Language model
- Full VLM integration
- Parameter counts
- Gradient flow
"""

import pytest
import torch
import torch.nn as nn
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
)
from src.models.vision_encoder import (
    PatchEmbed,
    ViTAttention,
    ViTMLP,
    ViTBlock,
    SigLIPVisionEncoder,
)
from src.models.projector import MlpProjector
from src.models.mla import (
    RMSNorm,
    RotaryEmbedding,
    MultiHeadLatentAttention,
    rotate_half,
    apply_rotary_pos_emb,
)
from src.models.moe import (
    MoEGate,
    MoEExpert,
    MoELayer,
    DenseFFN,
)
from src.models.language_model import DecoderLayer, DeepSeekMoELanguageModel
from src.models.deepseek_vl2 import DeepSeekVL2ForPretraining


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get test device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def vision_config():
    """Create test vision config (smaller for speed)."""
    return VisionConfig(
        image_size=224,
        patch_size=16,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        mlp_ratio=4,
    )


@pytest.fixture
def projector_config():
    """Create test projector config."""
    return ProjectorConfig(
        type="downsample_mlp_gelu",
        input_dim=256,
        output_dim=512,
        depth=2,
        downsample_ratio=2,
    )


@pytest.fixture
def language_config():
    """Create test language config (smaller for speed)."""
    return LanguageConfig(
        hidden_size=512,
        num_layers=4,
        num_heads=8,
        num_kv_heads=8,
        head_dim=64,
        intermediate_size=1024,
        vocab_size=1000,
        max_position_embeddings=512,
        mla=MLAConfig(
            q_lora_rank=256,
            kv_lora_rank=128,
            qk_rope_head_dim=32,
            qk_nope_head_dim=32,
            v_head_dim=64,
        ),
        moe=MoEConfig(
            enabled=True,
            num_experts=4,
            num_experts_per_token=2,
            expert_hidden_size=256,
            shared_expert_hidden_size=512,
            use_shared_expert=True,
        ),
    )


@pytest.fixture
def model_config(vision_config, projector_config, language_config):
    """Create test full model config."""
    return ModelConfig(
        variant="tiny",
        vision=vision_config,
        projector=projector_config,
        language=language_config,
    )


# =============================================================================
# Vision Encoder Tests
# =============================================================================

class TestPatchEmbed:
    """Test patch embedding layer."""
    
    def test_output_shape(self):
        """Test patch embed produces correct output shape."""
        patch_embed = PatchEmbed(img_size=224, patch_size=16, embed_dim=256)
        x = torch.randn(2, 3, 224, 224)
        out = patch_embed(x)
        
        expected_patches = (224 // 16) ** 2  # 196
        assert out.shape == (2, expected_patches, 256)
    
    def test_num_patches_calculation(self):
        """Test num_patches is calculated correctly."""
        patch_embed = PatchEmbed(img_size=384, patch_size=16, embed_dim=1024)
        expected = (384 // 16) ** 2  # 576
        assert patch_embed.num_patches == expected
    
    def test_different_image_sizes(self):
        """Test with different image sizes."""
        for img_size in [224, 384, 512]:
            patch_embed = PatchEmbed(img_size=img_size, patch_size=16, embed_dim=256)
            x = torch.randn(1, 3, img_size, img_size)
            out = patch_embed(x)
            expected_patches = (img_size // 16) ** 2
            assert out.shape == (1, expected_patches, 256)


class TestViTAttention:
    """Test Vision Transformer attention."""
    
    def test_output_shape(self):
        """Test attention output shape matches input."""
        attn = ViTAttention(dim=256, num_heads=4)
        x = torch.randn(2, 196, 256)
        out = attn(x)
        assert out.shape == x.shape
    
    def test_attention_scale(self):
        """Test attention scaling is correct."""
        attn = ViTAttention(dim=256, num_heads=4)
        expected_scale = (256 // 4) ** -0.5
        assert abs(attn.scale - expected_scale) < 1e-6


class TestViTMLP:
    """Test Vision Transformer MLP."""
    
    def test_output_shape(self):
        """Test MLP output shape matches input."""
        mlp = ViTMLP(dim=256, mlp_ratio=4)
        x = torch.randn(2, 196, 256)
        out = mlp(x)
        assert out.shape == x.shape
    
    def test_hidden_dim(self):
        """Test hidden dimension is scaled correctly."""
        mlp = ViTMLP(dim=256, mlp_ratio=4)
        assert mlp.fc1.out_features == 256 * 4
        assert mlp.fc2.in_features == 256 * 4


class TestSigLIPVisionEncoder:
    """Test full vision encoder."""
    
    def test_output_shape(self, vision_config):
        """Test encoder output shape."""
        encoder = SigLIPVisionEncoder(vision_config)
        x = torch.randn(2, 3, vision_config.image_size, vision_config.image_size)
        out = encoder(x)
        
        expected_patches = (vision_config.image_size // vision_config.patch_size) ** 2
        assert out.shape == (2, expected_patches, vision_config.hidden_size)
    
    def test_output_dim_property(self, vision_config):
        """Test output_dim property."""
        encoder = SigLIPVisionEncoder(vision_config)
        assert encoder.output_dim == vision_config.hidden_size
    
    def test_position_embedding_shape(self, vision_config):
        """Test position embedding has correct shape."""
        encoder = SigLIPVisionEncoder(vision_config)
        expected_patches = (vision_config.image_size // vision_config.patch_size) ** 2
        assert encoder.pos_embed.shape == (1, expected_patches, vision_config.hidden_size)
    
    def test_num_layers(self, vision_config):
        """Test correct number of transformer blocks."""
        encoder = SigLIPVisionEncoder(vision_config)
        assert len(encoder.blocks) == vision_config.num_layers


# =============================================================================
# MLA Tests
# =============================================================================

class TestRMSNorm:
    """Test RMS normalization."""
    
    def test_output_shape(self):
        """Test RMSNorm preserves shape."""
        norm = RMSNorm(256)
        x = torch.randn(2, 10, 256)
        out = norm(x)
        assert out.shape == x.shape
    
    def test_normalization(self):
        """Test values are normalized."""
        norm = RMSNorm(256)
        x = torch.randn(2, 10, 256) * 100  # Large values
        out = norm(x)
        # Output should have reasonable magnitude
        assert out.abs().mean() < 10


class TestRotaryEmbedding:
    """Test Rotary Position Embedding."""
    
    def test_cos_sin_shapes(self):
        """Test cos and sin have correct shapes."""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
        x = torch.randn(2, 10, 4, 64)
        position_ids = torch.arange(10).unsqueeze(0)
        
        cos, sin = rope(x, position_ids)
        assert cos.shape == (1, 10, 1, 64)
        assert sin.shape == (1, 10, 1, 64)
    
    def test_rotate_half(self):
        """Test rotate_half function."""
        x = torch.randn(2, 10, 4, 64)
        rotated = rotate_half(x)
        assert rotated.shape == x.shape


class TestMultiHeadLatentAttention:
    """Test Multi-head Latent Attention."""
    
    def test_output_shape(self, language_config):
        """Test MLA output shape."""
        mla = MultiHeadLatentAttention(language_config, layer_idx=0)
        x = torch.randn(2, 32, language_config.hidden_size)
        
        out, _ = mla(x)
        assert out.shape == x.shape
    
    def test_kv_cache(self, language_config):
        """Test KV cache is returned when use_cache=True."""
        mla = MultiHeadLatentAttention(language_config, layer_idx=0)
        x = torch.randn(2, 32, language_config.hidden_size)
        
        out, past_kv = mla(x, use_cache=True)
        assert past_kv is not None
        assert len(past_kv) == 2  # (k, v)
    
    def test_incremental_decoding(self, language_config):
        """Test incremental decoding with KV cache."""
        mla = MultiHeadLatentAttention(language_config, layer_idx=0)
        
        # First pass
        x1 = torch.randn(2, 32, language_config.hidden_size)
        out1, past_kv = mla(x1, use_cache=True)
        
        # Second pass with cache
        x2 = torch.randn(2, 1, language_config.hidden_size)
        out2, past_kv2 = mla(x2, past_key_value=past_kv, use_cache=True)
        
        assert out2.shape == (2, 1, language_config.hidden_size)
        assert past_kv2[0].shape[2] == 33  # 32 + 1


# =============================================================================
# MoE Tests
# =============================================================================

class TestMoEGate:
    """Test MoE routing gate."""
    
    def test_routing_shape(self):
        """Test routing outputs have correct shapes."""
        gate = MoEGate(hidden_size=256, num_experts=8, num_experts_per_token=2)
        x = torch.randn(64, 256)  # [batch*seq, hidden]
        
        weights, indices, logits = gate(x)
        
        assert weights.shape == (64, 2)  # top-k weights
        assert indices.shape == (64, 2)  # top-k indices
        assert logits.shape == (64, 8)  # all expert logits
    
    def test_weights_normalized(self):
        """Test routing weights sum to 1."""
        gate = MoEGate(hidden_size=256, num_experts=8, num_experts_per_token=2)
        x = torch.randn(64, 256)
        
        weights, _, _ = gate(x)
        sums = weights.sum(dim=-1)
        
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_indices_in_range(self):
        """Test indices are within valid range."""
        num_experts = 8
        gate = MoEGate(hidden_size=256, num_experts=num_experts, num_experts_per_token=2)
        x = torch.randn(64, 256)
        
        _, indices, _ = gate(x)
        
        assert indices.min() >= 0
        assert indices.max() < num_experts


class TestMoEExpert:
    """Test individual MoE expert."""
    
    def test_output_shape(self):
        """Test expert output shape."""
        expert = MoEExpert(hidden_size=256, intermediate_size=512)
        x = torch.randn(32, 256)
        
        out = expert(x)
        assert out.shape == x.shape
    
    def test_swiglu_activation(self):
        """Test SwiGLU-style activation is used."""
        expert = MoEExpert(hidden_size=256, intermediate_size=512)
        assert hasattr(expert, 'gate_proj')
        assert hasattr(expert, 'up_proj')
        assert hasattr(expert, 'down_proj')


class TestMoELayer:
    """Test full MoE layer."""
    
    def test_output_shape(self, language_config):
        """Test MoE layer output shape."""
        moe = MoELayer(language_config)
        x = torch.randn(2, 32, language_config.hidden_size)
        
        out = moe(x)
        assert out.shape == x.shape
    
    def test_shared_expert(self, language_config):
        """Test shared expert is used when configured."""
        moe = MoELayer(language_config)
        assert moe.use_shared_expert == True
        assert hasattr(moe, 'shared_expert')
        assert hasattr(moe, 'shared_expert_gate')
    
    def test_num_experts(self, language_config):
        """Test correct number of experts created."""
        moe = MoELayer(language_config)
        assert len(moe.experts) == language_config.moe.num_experts
    
    def test_router_logits(self, language_config):
        """Test router logits are returned when requested."""
        moe = MoELayer(language_config)
        x = torch.randn(2, 32, language_config.hidden_size)
        
        out, logits = moe(x, return_router_logits=True)
        assert logits is not None


class TestDenseFFN:
    """Test dense FFN (non-MoE)."""
    
    def test_output_shape(self):
        """Test dense FFN output shape."""
        ffn = DenseFFN(hidden_size=256, intermediate_size=512)
        x = torch.randn(2, 32, 256)
        
        out = ffn(x)
        assert out.shape == x.shape


# =============================================================================
# Projector Tests
# =============================================================================

class TestMlpProjector:
    """Test VL projector."""
    
    def test_downsample_mlp_shape(self, projector_config):
        """Test downsampling projector output shape."""
        projector = MlpProjector(projector_config)
        # Input: 196 patches from 224x224 image with patch_size=16
        x = torch.randn(2, 196, 256)
        
        out = projector(x)
        # With 2x downsample: 196/4 = 49
        assert out.shape == (2, 49, 512)
    
    def test_linear_projector(self):
        """Test simple linear projector."""
        config = ProjectorConfig(
            type="linear",
            input_dim=256,
            output_dim=512,
        )
        projector = MlpProjector(config)
        x = torch.randn(2, 196, 256)
        
        out = projector(x)
        assert out.shape == (2, 196, 512)
    
    def test_output_dim_property(self, projector_config):
        """Test output_dim property."""
        projector = MlpProjector(projector_config)
        assert projector.output_dim == projector_config.output_dim


# =============================================================================
# Language Model Tests
# =============================================================================

class TestDecoderLayer:
    """Test decoder layer."""
    
    def test_output_shape(self, language_config):
        """Test decoder layer output shape."""
        layer = DecoderLayer(language_config, layer_idx=0)
        x = torch.randn(2, 32, language_config.hidden_size)
        
        out, _ = layer(x)
        assert out.shape == x.shape
    
    def test_moe_layer_selection(self, language_config):
        """Test MoE vs dense layer selection."""
        # First layer should be MoE (layer_freq=1)
        layer0 = DecoderLayer(language_config, layer_idx=0)
        assert layer0.is_moe == True
        
        # Disable MoE to test dense
        language_config.moe.enabled = False
        layer1 = DecoderLayer(language_config, layer_idx=0)
        assert layer1.is_moe == False


class TestDeepSeekMoELanguageModel:
    """Test full language model."""
    
    def test_output_shape(self, language_config):
        """Test LM output shape."""
        lm = DeepSeekMoELanguageModel(language_config)
        input_ids = torch.randint(0, language_config.vocab_size, (2, 32))
        
        outputs = lm(input_ids=input_ids)
        hidden_states = outputs[0]
        
        assert hidden_states.shape == (2, 32, language_config.hidden_size)
    
    def test_inputs_embeds(self, language_config):
        """Test using inputs_embeds instead of input_ids."""
        lm = DeepSeekMoELanguageModel(language_config)
        inputs_embeds = torch.randn(2, 32, language_config.hidden_size)
        
        outputs = lm(inputs_embeds=inputs_embeds)
        assert outputs[0].shape == inputs_embeds.shape
    
    def test_kv_cache(self, language_config):
        """Test KV cache."""
        lm = DeepSeekMoELanguageModel(language_config)
        input_ids = torch.randint(0, language_config.vocab_size, (2, 32))
        
        outputs = lm(input_ids=input_ids, use_cache=True)
        past_kv = outputs[1]
        
        assert past_kv is not None
        assert len(past_kv) == language_config.num_layers
    
    def test_num_layers(self, language_config):
        """Test correct number of layers."""
        lm = DeepSeekMoELanguageModel(language_config)
        assert len(lm.layers) == language_config.num_layers


# =============================================================================
# Full Model Tests
# =============================================================================

class TestDeepSeekVL2ForPretraining:
    """Test full VLM model."""
    
    def test_model_creation(self, model_config):
        """Test model can be created."""
        model = DeepSeekVL2ForPretraining(model_config)
        assert model is not None
    
    def test_text_only_forward(self, model_config):
        """Test forward pass with text only."""
        model = DeepSeekVL2ForPretraining(model_config)
        input_ids = torch.randint(0, model_config.language.vocab_size, (2, 32))
        labels = input_ids.clone()
        
        outputs = model(input_ids=input_ids, labels=labels)
        
        assert "loss" in outputs
        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 32, model_config.language.vocab_size)
    
    def test_image_encoding(self, model_config):
        """Test image encoding."""
        model = DeepSeekVL2ForPretraining(model_config)
        images = torch.randn(2, 3, model_config.vision.image_size, model_config.vision.image_size)
        
        image_embeds = model.encode_images(images)
        assert image_embeds.dim() == 3
        assert image_embeds.shape[0] == 2
    
    def test_multimodal_forward(self, model_config):
        """Test forward pass with images and text."""
        model = DeepSeekVL2ForPretraining(model_config)
        
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, model_config.language.vocab_size, (batch_size, seq_len))
        images = torch.randn(batch_size, 3, model_config.vision.image_size, model_config.vision.image_size)
        image_positions = torch.tensor([[16], [16]])
        labels = input_ids.clone()
        
        outputs = model(
            input_ids=input_ids,
            images=images,
            image_positions=image_positions,
            labels=labels,
        )
        
        assert outputs["loss"] is not None
        assert outputs["logits"].shape == (batch_size, seq_len, model_config.language.vocab_size)
    
    def test_freeze_unfreeze_vision(self, model_config):
        """Test freezing/unfreezing vision encoder."""
        model = DeepSeekVL2ForPretraining(model_config)
        
        # Freeze
        model.freeze_vision_encoder()
        for param in model.vision_encoder.parameters():
            assert param.requires_grad == False
        
        # Unfreeze
        model.unfreeze_vision_encoder()
        for param in model.vision_encoder.parameters():
            assert param.requires_grad == True
    
    def test_freeze_unfreeze_lm(self, model_config):
        """Test freezing/unfreezing language model."""
        model = DeepSeekVL2ForPretraining(model_config)
        
        # Freeze
        model.freeze_language_model()
        for param in model.language_model.parameters():
            assert param.requires_grad == False
        
        # Unfreeze
        model.unfreeze_language_model()
        for param in model.language_model.parameters():
            assert param.requires_grad == True
    
    def test_gradient_checkpointing(self, model_config):
        """Test gradient checkpointing can be enabled."""
        model = DeepSeekVL2ForPretraining(model_config)
        
        model.enable_gradient_checkpointing()
        assert model.gradient_checkpointing == True
        assert model.language_model.gradient_checkpointing == True
        
        model.disable_gradient_checkpointing()
        assert model.gradient_checkpointing == False


# =============================================================================
# Parameter Count Tests
# =============================================================================

class TestParameterCounts:
    """Test parameter counting."""
    
    def test_vision_encoder_params(self, vision_config):
        """Test vision encoder parameter count is reasonable."""
        encoder = SigLIPVisionEncoder(vision_config)
        params = sum(p.numel() for p in encoder.parameters())
        
        # Should have > 0 params
        assert params > 0
        # For small config, should be < 100M
        assert params < 100_000_000
    
    def test_moe_params(self, language_config):
        """Test MoE has more params than dense equivalent."""
        moe = MoELayer(language_config)
        moe_params = sum(p.numel() for p in moe.parameters())
        
        dense = DenseFFN(language_config.hidden_size, language_config.intermediate_size)
        dense_params = sum(p.numel() for p in dense.parameters())
        
        # MoE should have more params due to multiple experts
        assert moe_params > dense_params
    
    def test_full_model_params(self, model_config):
        """Test full model parameter count."""
        model = DeepSeekVL2ForPretraining(model_config)
        
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total > 0
        assert trainable == total  # All should be trainable initially


# =============================================================================
# Gradient Flow Tests
# =============================================================================

class TestGradientFlow:
    """Test gradients flow correctly through model."""
    
    def test_vision_encoder_gradients(self, vision_config, device):
        """Test gradients flow through vision encoder."""
        encoder = SigLIPVisionEncoder(vision_config).to(device)
        x = torch.randn(2, 3, vision_config.image_size, vision_config.image_size, device=device)
        
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    
    def test_mla_gradients(self, language_config, device):
        """Test gradients flow through MLA."""
        mla = MultiHeadLatentAttention(language_config).to(device)
        x = torch.randn(2, 32, language_config.hidden_size, device=device)
        
        out, _ = mla(x)
        loss = out.sum()
        loss.backward()
        
        # Check key projections have gradients
        assert mla.q_a_proj.weight.grad is not None
        assert mla.kv_a_proj.weight.grad is not None
        assert not torch.isnan(mla.q_a_proj.weight.grad).any()
    
    def test_moe_gradients(self, language_config, device):
        """Test gradients flow through MoE."""
        moe = MoELayer(language_config).to(device)
        x = torch.randn(2, 32, language_config.hidden_size, device=device)
        
        out = moe(x)
        loss = out.sum()
        loss.backward()
        
        # Check at least some experts have gradients
        experts_with_grad = sum(
            1 for expert in moe.experts
            if expert.gate_proj.weight.grad is not None
        )
        assert experts_with_grad > 0
    
    def test_full_model_gradients(self, model_config, device):
        """Test gradients flow through full model."""
        model = DeepSeekVL2ForPretraining(model_config).to(device)
        
        input_ids = torch.randint(0, model_config.language.vocab_size, (2, 32), device=device)
        labels = input_ids.clone()
        
        outputs = model(input_ids=input_ids, labels=labels)
        outputs["loss"].backward()
        
        # Check key components have gradients
        assert model.vision_encoder.patch_embed.proj.weight.grad is not None
        assert model.language_model.embed_tokens.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
