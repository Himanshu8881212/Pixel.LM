"""
Tests for DeepSeek V3 innovations: aux-loss-free training and NaN prevention.

Tests:
1. Aux-loss-free load balancing (dynamic bias)
2. FP32 accumulation in MoE
3. Expert bias updates
4. NaN prevention in edge cases
5. Numerical stability with extreme values
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    LanguageConfig,
    ModelConfig,
    VisionConfig,
    ProjectorConfig,
    MLAConfig,
    MoEConfig,
)
from src.models.moe import MoELayer, MoEGate, MoEExpert
from src.models.deepseek_vl2 import DeepSeekVL2ForPretraining


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Aux-Loss-Free Load Balancing Tests
# =============================================================================

class TestAuxLossFree:
    """Test aux-loss-free load balancing mechanism."""
    
    def test_expert_bias_initialized_zero(self, device):
        """Test expert bias starts at zero."""
        config = LanguageConfig(
            hidden_size=256,
            moe=MoEConfig(enabled=True, num_experts=8, aux_loss_alpha=0.0)
        )
        moe = MoELayer(config).to(device)
        
        assert torch.allclose(moe.gate.expert_bias, torch.zeros(8, device=device))
    
    def test_expert_bias_updates_during_training(self, device):
        """Test expert bias updates dynamically during training."""
        config = LanguageConfig(
            hidden_size=256,
            moe=MoEConfig(enabled=True, num_experts=8, aux_loss_alpha=0.0)
        )
        moe = MoELayer(config).to(device)
        moe.train()
        
        # Store initial bias
        initial_bias = moe.gate.expert_bias.clone()
        
        # Run forward pass multiple times
        for _ in range(10):
            x = torch.randn(4, 32, 256, device=device)
            _ = moe(x)
        
        # Bias should have changed
        assert not torch.allclose(moe.gate.expert_bias, initial_bias)
    
    def test_expert_bias_not_updated_during_eval(self, device):
        """Test expert bias doesn't update during evaluation."""
        config = LanguageConfig(
            hidden_size=256,
            moe=MoEConfig(enabled=True, num_experts=8, aux_loss_alpha=0.0)
        )
        moe = MoELayer(config).to(device)
        moe.eval()
        
        initial_bias = moe.gate.expert_bias.clone()
        
        for _ in range(10):
            x = torch.randn(4, 32, 256, device=device)
            _ = moe(x)
        
        # Bias should NOT have changed
        assert torch.allclose(moe.gate.expert_bias, initial_bias)
    
    def test_aux_loss_is_zero_when_disabled(self, device):
        """Test aux loss returns 0 when aux_loss_alpha=0."""
        config = LanguageConfig(
            hidden_size=256,
            moe=MoEConfig(enabled=True, num_experts=8, aux_loss_alpha=0.0)
        )
        moe = MoELayer(config).to(device)
        
        x = torch.randn(4, 32, 256, device=device)
        _, router_logits = moe(x, return_router_logits=True)
        
        aux_loss = moe.compute_aux_loss(router_logits)
        assert aux_loss.item() == 0.0
    
    def test_expert_bias_bounded(self, device):
        """Test expert bias stays within bounds."""
        config = LanguageConfig(
            hidden_size=256,
            moe=MoEConfig(enabled=True, num_experts=8, aux_loss_alpha=0.0)
        )
        moe = MoELayer(config).to(device)
        moe.train()
        
        # Run many forward passes
        for _ in range(100):
            x = torch.randn(4, 32, 256, device=device)
            _ = moe(x)
        
        # Bias should be bounded
        assert moe.gate.expert_bias.abs().max() <= 1.0


# =============================================================================
# FP32 Accumulation Tests
# =============================================================================

class TestFP32Accumulation:
    """Test FP32 accumulation for numerical stability."""
    
    def test_moe_output_finite_with_bf16_input(self, device):
        """Test MoE produces finite output with bf16 input."""
        if device == "cpu":
            pytest.skip("BF16 testing on CUDA")
        
        config = LanguageConfig(
            hidden_size=256,
            moe=MoEConfig(enabled=True, num_experts=8)
        )
        moe = MoELayer(config).to(device).to(torch.bfloat16)
        
        x = torch.randn(4, 32, 256, device=device, dtype=torch.bfloat16)
        out = moe(x)
        
        assert torch.isfinite(out).all()
    
    def test_moe_output_finite_with_extreme_values(self, device):
        """Test MoE handles extreme input values."""
        config = LanguageConfig(
            hidden_size=256,
            moe=MoEConfig(enabled=True, num_experts=8)
        )
        moe = MoELayer(config).to(device)
        
        # Large values that could cause overflow
        x = torch.randn(4, 32, 256, device=device) * 100
        out = moe(x)
        
        assert torch.isfinite(out).all()
    
    def test_moe_output_finite_with_small_values(self, device):
        """Test MoE handles very small input values."""
        config = LanguageConfig(
            hidden_size=256,
            moe=MoEConfig(enabled=True, num_experts=8)
        )
        moe = MoELayer(config).to(device)
        
        # Very small values that could cause underflow
        x = torch.randn(4, 32, 256, device=device) * 1e-10
        out = moe(x)
        
        assert torch.isfinite(out).all()
    
    def test_accumulated_output_matches_dtype(self, device):
        """Test output dtype matches input dtype."""
        config = LanguageConfig(
            hidden_size=256,
            moe=MoEConfig(enabled=True, num_experts=4)
        )
        moe = MoELayer(config).to(device)
        
        # FP32 input
        x_fp32 = torch.randn(4, 32, 256, device=device, dtype=torch.float32)
        out_fp32 = moe(x_fp32)
        assert out_fp32.dtype == torch.float32
        
        # FP16 input (if supported)
        if device == "cuda":
            x_fp16 = torch.randn(4, 32, 256, device=device, dtype=torch.float16)
            out_fp16 = moe(x_fp16)
            assert out_fp16.dtype == torch.float16


# =============================================================================
# Expert Load Balancing Tests
# =============================================================================

class TestExpertLoadBalancing:
    """Test expert load balancing behavior."""
    
    def test_all_experts_receive_tokens(self, device):
        """Test all experts receive some tokens over time."""
        config = LanguageConfig(
            hidden_size=256,
            moe=MoEConfig(enabled=True, num_experts=8, num_experts_per_token=2)
        )
        moe = MoELayer(config).to(device)
        moe.train()
        
        expert_activated = torch.zeros(8, device=device)
        
        for _ in range(50):
            x = torch.randn(8, 64, 256, device=device)
            _, router_logits = moe(x, return_router_logits=True)
            
            # Track which experts are activated
            probs = torch.softmax(router_logits, dim=-1)
            _, indices = torch.topk(probs, 2, dim=-1)
            
            for i in range(8):
                if (indices == i).any():
                    expert_activated[i] = 1.0
        
        # All experts should be activated at some point
        assert expert_activated.sum() == 8
    
    def test_expert_load_ema_tracks_usage(self, device):
        """Test expert load EMA tracks actual usage."""
        config = LanguageConfig(
            hidden_size=256,
            moe=MoEConfig(enabled=True, num_experts=4)
        )
        moe = MoELayer(config).to(device)
        moe.train()
        
        # Initial EMA should be uniform
        assert torch.allclose(
            moe.gate.expert_load_ema,
            torch.ones(4, device=device) / 4
        )
        
        # After training, EMA should reflect actual load
        for _ in range(100):
            x = torch.randn(4, 32, 256, device=device)
            _ = moe(x)
        
        # EMA should have been updated
        assert not torch.allclose(
            moe.gate.expert_load_ema,
            torch.ones(4, device=device) / 4
        )


# =============================================================================
# NaN Prevention Tests
# =============================================================================

class TestNaNPrevention:
    """Test NaN prevention mechanisms."""
    
    def test_no_nan_with_zero_input(self, device):
        """Test no NaN with zero input."""
        config = LanguageConfig(
            hidden_size=256,
            moe=MoEConfig(enabled=True, num_experts=8)
        )
        moe = MoELayer(config).to(device)
        
        x = torch.zeros(4, 32, 256, device=device)
        out = moe(x)
        
        assert torch.isfinite(out).all()
    
    def test_no_nan_in_gradient(self, device):
        """Test no NaN in gradients."""
        config = LanguageConfig(
            hidden_size=256,
            moe=MoEConfig(enabled=True, num_experts=4)
        )
        moe = MoELayer(config).to(device)
        moe.train()
        
        x = torch.randn(4, 32, 256, device=device, requires_grad=True)
        out = moe(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        
        for name, param in moe.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"NaN in {name}.grad"
    
    def test_router_softmax_no_overflow(self, device):
        """Test router softmax doesn't overflow with large logits."""
        gate = MoEGate(
            hidden_size=256,
            num_experts=8,
            num_experts_per_token=2
        ).to(device)
        
        # Large input that could cause large logits
        x = torch.randn(64, 256, device=device) * 50
        weights, indices, logits = gate(x)
        
        assert torch.isfinite(weights).all()
        assert torch.isfinite(logits).all()
    
    def test_weight_normalization_no_divide_by_zero(self, device):
        """Test weight normalization handles edge cases."""
        gate = MoEGate(
            hidden_size=256,
            num_experts=8,
            num_experts_per_token=2
        ).to(device)
        
        # Near-zero input could lead to near-zero weights
        x = torch.randn(64, 256, device=device) * 1e-10
        weights, indices, logits = gate(x)
        
        # Weights should sum to 1 (not NaN)
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


# =============================================================================
# Full Model Integration Tests
# =============================================================================

class TestFullModelIntegration:
    """Test aux-loss-free training in full model."""
    
    def test_full_model_forward_with_moe(self, device):
        """Test full model forward pass with MoE."""
        config = ModelConfig(
            vision=VisionConfig(image_size=56, hidden_size=64, num_layers=1, num_heads=2, patch_size=8),
            projector=ProjectorConfig(input_dim=64, output_dim=128, type="linear"),
            language=LanguageConfig(
                hidden_size=128, num_layers=2, vocab_size=256,
                mla=MLAConfig(q_lora_rank=64, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16),
                moe=MoEConfig(enabled=True, num_experts=4, num_experts_per_token=2, aux_loss_alpha=0.0)
            ),
        )
        model = DeepSeekVL2ForPretraining(config).to(device)
        
        input_ids = torch.randint(0, 256, (2, 16), device=device)
        outputs = model(input_ids=input_ids)
        
        assert torch.isfinite(outputs["logits"]).all()
    
    def test_training_step_with_aux_loss_free(self, device):
        """Test training step with aux-loss-free MoE."""
        config = ModelConfig(
            vision=VisionConfig(image_size=56, hidden_size=64, num_layers=1, num_heads=2, patch_size=8),
            projector=ProjectorConfig(input_dim=64, output_dim=128, type="linear"),
            language=LanguageConfig(
                hidden_size=128, num_layers=2, vocab_size=256,
                mla=MLAConfig(q_lora_rank=64, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16),
                moe=MoEConfig(enabled=True, num_experts=4, layer_freq=2, aux_loss_alpha=0.0)
            ),
        )
        model = DeepSeekVL2ForPretraining(config).to(device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        input_ids = torch.randint(0, 256, (2, 16), device=device)
        labels = input_ids.clone()
        
        # Multiple training steps
        for _ in range(5):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            
            assert torch.isfinite(loss)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
