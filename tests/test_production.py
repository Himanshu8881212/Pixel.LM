"""
Production Feature Tests for PixelLM.

Tests for:
  - Flash Attention 3 (with fallback)
  - FP8 utilities
  - DeepSpeed configuration
  - Gradient checkpointing
  - Expert Parallelism
"""

import pytest
import torch
import torch.nn as nn
import json
from pathlib import Path


# =============================================================================
# Flash Attention Tests
# =============================================================================

class TestFlashAttention:
    """Test Flash Attention 3 integration."""
    
    def test_flash_attn_import(self):
        """Test Flash Attention import and flag."""
        from src.model import FLASH_ATTN_AVAILABLE
        # Should be either True or False (no import error)
        assert isinstance(FLASH_ATTN_AVAILABLE, bool)
    
    def test_mla_with_flash(self):
        """Test MLA forward pass with Flash Attention."""
        from src.model import MLA, LanguageConfig, MLAConfig
        
        cfg = LanguageConfig(
            hidden_size=256,
            num_heads=4,
            head_dim=64,
            mla=MLAConfig(
                q_lora_rank=128,
                kv_lora_rank=64,
                qk_rope_head_dim=32,
                qk_nope_head_dim=32,
                v_head_dim=32,
            ),
        )
        
        mla = MLA(cfg, layer_idx=0)
        x = torch.randn(2, 16, 256)
        
        # Test with flash=True (should fallback gracefully if unavailable)
        out, _ = mla(x, use_flash=True)
        assert out.shape == x.shape
        
        # Test with flash=False (explicit SDPA)
        out, _ = mla(x, use_flash=False)
        assert out.shape == x.shape
    
    def test_mla_cache(self):
        """Test MLA with KV cache."""
        from src.model import MLA, LanguageConfig, MLAConfig
        
        cfg = LanguageConfig(
            hidden_size=256,
            num_heads=4,
            head_dim=64,
            mla=MLAConfig(
                q_lora_rank=128,
                kv_lora_rank=64,
                qk_rope_head_dim=32,
                qk_nope_head_dim=32,
                v_head_dim=32,
            ),
        )
        
        mla = MLA(cfg, layer_idx=0)
        
        # First pass
        x1 = torch.randn(2, 8, 256)
        out1, kv1 = mla(x1, use_cache=True)
        assert kv1 is not None
        
        # Second pass with cache
        x2 = torch.randn(2, 1, 256)
        out2, kv2 = mla(x2, past_kv=kv1, use_cache=True)
        assert out2.shape == (2, 1, 256)


# =============================================================================
# FP8 Utilities Tests
# =============================================================================

class TestFP8Utils:
    """Test FP8 training utilities."""
    
    def test_fp8_config(self):
        """Test FP8Config dataclass."""
        from src.fp8_utils import FP8Config
        
        config = FP8Config(enabled=True, margin=1, interval=2)
        assert config.enabled == True
        assert config.margin == 1
        assert config.interval == 2
    
    def test_te_available_flag(self):
        """Test TransformerEngine availability flag."""
        from src.fp8_utils import TE_AVAILABLE
        assert isinstance(TE_AVAILABLE, bool)
    
    def test_flash_attn_flags(self):
        """Test Flash Attention availability flags."""
        from src.fp8_utils import FLASH_ATTN_AVAILABLE, FLASH_ATTN_3
        assert isinstance(FLASH_ATTN_AVAILABLE, bool)
        assert isinstance(FLASH_ATTN_3, bool)
    
    def test_flash_attention_wrapper(self):
        """Test flash_attention function wrapper."""
        from src.fp8_utils import flash_attention
        
        B, H, L, D = 2, 4, 16, 32
        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)
        
        out = flash_attention(q, k, v, causal=True)
        assert out.shape == (B, H, L, D)
    
    def test_fp8_autocast(self):
        """Test fp8_autocast context manager."""
        from src.fp8_utils import fp8_autocast
        
        # Should not raise even without TransformerEngine
        with fp8_autocast(enabled=False):
            x = torch.randn(2, 2)
            y = x + x
            assert y.shape == x.shape


# =============================================================================
# DeepSpeed Config Tests
# =============================================================================

class TestDeepSpeedConfig:
    """Test DeepSpeed configuration files."""
    
    @pytest.fixture
    def config_dir(self):
        return Path(__file__).parent.parent / "configs"
    
    def test_zero2_config_exists(self, config_dir):
        """Test ZeRO-2 config file exists."""
        config_path = config_dir / "ds_zero2.json"
        assert config_path.exists(), f"Missing {config_path}"
    
    def test_zero3_config_exists(self, config_dir):
        """Test ZeRO-3 config file exists."""
        config_path = config_dir / "ds_zero3.json"
        assert config_path.exists(), f"Missing {config_path}"
    
    def test_zero2_config_valid(self, config_dir):
        """Test ZeRO-2 config is valid JSON."""
        config_path = config_dir / "ds_zero2.json"
        with open(config_path) as f:
            config = json.load(f)
        
        assert "zero_optimization" in config
        assert config["zero_optimization"]["stage"] == 2
        assert config["bf16"]["enabled"] == True
    
    def test_zero3_config_valid(self, config_dir):
        """Test ZeRO-3 config is valid JSON."""
        config_path = config_dir / "ds_zero3.json"
        with open(config_path) as f:
            config = json.load(f)
        
        assert "zero_optimization" in config
        assert config["zero_optimization"]["stage"] == 3
        assert "activation_checkpointing" in config


# =============================================================================
# Gradient Checkpointing Tests
# =============================================================================

class TestGradientCheckpointing:
    """Test gradient checkpointing functionality."""
    
    def test_language_model_checkpointing_flag(self):
        """Test LanguageModel has gradient_checkpointing attribute."""
        from src.model import LanguageModel, LanguageConfig
        
        cfg = LanguageConfig(
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            intermediate_size=512,
        )
        cfg.moe.enabled = False
        
        lm = LanguageModel(cfg)
        assert hasattr(lm, 'gradient_checkpointing')
        assert lm.gradient_checkpointing == False
    
    def test_gradient_checkpointing_enabled(self):
        """Test gradient checkpointing can be enabled."""
        from src.model import LanguageModel, LanguageConfig
        
        cfg = LanguageConfig(
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            intermediate_size=512,
        )
        cfg.moe.enabled = False
        
        lm = LanguageModel(cfg)
        lm.gradient_checkpointing = True
        
        x = torch.randint(0, 1000, (2, 8))
        lm.train()
        
        # Should work with checkpointing enabled
        out, _ = lm(x)
        assert out.shape == (2, 8, 256)
    
    def test_checkpointing_memory_savings(self):
        """Test that checkpointing reduces memory during training."""
        from src.model import LanguageModel, LanguageConfig
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        cfg = LanguageConfig(
            hidden_size=512,
            num_layers=4,
            num_heads=8,
            head_dim=64,
            intermediate_size=1024,
        )
        cfg.moe.enabled = False
        
        device = torch.device("cuda")
        
        # Measure without checkpointing
        lm1 = LanguageModel(cfg).to(device)
        lm1.gradient_checkpointing = False
        lm1.train()
        
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, 1000, (4, 64), device=device)
        out = lm1(x)[0]
        loss = out.sum()
        loss.backward()
        mem_no_ckpt = torch.cuda.max_memory_allocated()
        
        del lm1, out, loss
        torch.cuda.empty_cache()
        
        # Measure with checkpointing
        lm2 = LanguageModel(cfg).to(device)
        lm2.gradient_checkpointing = True
        lm2.train()
        
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(0, 1000, (4, 64), device=device)
        out = lm2(x)[0]
        loss = out.sum()
        loss.backward()
        mem_ckpt = torch.cuda.max_memory_allocated()
        
        # Checkpointing should use less peak memory
        assert mem_ckpt <= mem_no_ckpt


# =============================================================================
# Expert Parallelism Tests
# =============================================================================

class TestExpertParallelism:
    """Test Expert Parallelism support."""
    
    def test_deepspeed_moe_import(self):
        """Test DeepSpeed MoE import flag."""
        from src.model import DEEPSPEED_MOE_AVAILABLE
        assert isinstance(DEEPSPEED_MOE_AVAILABLE, bool)
    
    def test_moe_expert_count(self):
        """Test MoE has correct number of experts."""
        from src.model import MoE, LanguageConfig
        
        cfg = LanguageConfig(
            hidden_size=256,
            num_heads=4,
        )
        cfg.moe.num_experts = 16
        cfg.moe.num_experts_per_token = 2
        cfg.moe.expert_hidden_size = 128
        cfg.moe.shared_expert_hidden_size = 256
        
        moe = MoE(cfg)
        assert len(moe.experts) == 16
        assert moe.topk == 2
    
    def test_moe_forward(self):
        """Test MoE forward pass."""
        from src.model import MoE, LanguageConfig
        
        cfg = LanguageConfig(
            hidden_size=256,
            num_heads=4,
        )
        cfg.moe.num_experts = 8
        cfg.moe.num_experts_per_token = 2
        cfg.moe.expert_hidden_size = 128
        cfg.moe.shared_expert_hidden_size = 256
        
        moe = MoE(cfg)
        x = torch.randn(2, 16, 256)
        
        out = moe(x)
        assert out.shape == x.shape


# =============================================================================
# Integration Tests
# =============================================================================

class TestProductionIntegration:
    """Integration tests for production features."""
    
    def test_full_model_with_production_features(self):
        """Test full model with all production features."""
        from src.model import get_config, PixelLM
        
        cfg = get_config("tiny")
        model = PixelLM(cfg)
        
        # Enable gradient checkpointing
        model.language.gradient_checkpointing = True
        
        # Create dummy input
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32)
        
        model.train()
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        
        assert outputs.loss is not None
        assert outputs.logits.shape == (2, 32, 32000)
    
    def test_model_forward_backward(self):
        """Test model forward and backward pass."""
        from src.model import get_config, PixelLM
        
        cfg = get_config("tiny")
        model = PixelLM(cfg)
        model.train()
        
        images = torch.randn(2, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (2, 16))
        
        outputs = model(
            images=images,
            input_ids=input_ids,
            labels=input_ids,
        )
        
        # Backward pass should work
        outputs.loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None or param.numel() == 0, f"No gradient for {name}"
