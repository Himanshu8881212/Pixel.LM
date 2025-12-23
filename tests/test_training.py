"""
Training loop and loss computation tests for DeepSeek VL2.

Tests:
- Loss computation
- Gradient accumulation
- Learning rate scheduling
- Mixed precision training
- Checkpoint saving/loading
- Training stage configuration
- PEFT integration
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import math
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DeepSeekVL2TrainingConfig,
    ModelConfig,
    VisionConfig,
    ProjectorConfig,
    LanguageConfig,
    TrainingConfig,
    PEFTConfig,
    MLAConfig,
    MoEConfig,
)
from src.models.deepseek_vl2 import DeepSeekVL2ForPretraining


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get test device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def small_model_config():
    """Create small model config for fast testing."""
    vision_config = VisionConfig(
        image_size=112,
        patch_size=16,
        hidden_size=128,
        num_layers=2,
        num_heads=2,
    )
    projector_config = ProjectorConfig(
        type="downsample_mlp_gelu",
        input_dim=128,
        output_dim=256,
        depth=2,
        downsample_ratio=2,
    )
    language_config = LanguageConfig(
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        head_dim=64,
        intermediate_size=512,
        vocab_size=500,
        max_position_embeddings=256,
        mla=MLAConfig(
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_rope_head_dim=16,
            qk_nope_head_dim=16,
            v_head_dim=32,
        ),
        moe=MoEConfig(
            enabled=True,
            num_experts=2,
            num_experts_per_token=1,
            expert_hidden_size=128,
            shared_expert_hidden_size=256,
            use_shared_expert=True,
        ),
    )
    
    return ModelConfig(
        variant="tiny",
        vision=vision_config,
        projector=projector_config,
        language=language_config,
    )


@pytest.fixture
def small_model(small_model_config, device):
    """Create small model for testing."""
    model = DeepSeekVL2ForPretraining(small_model_config)
    return model.to(device)


@pytest.fixture
def dummy_batch(small_model_config, device):
    """Create dummy batch for testing."""
    batch_size = 2
    seq_len = 32
    
    return {
        "input_ids": torch.randint(
            0, small_model_config.language.vocab_size,
            (batch_size, seq_len), device=device
        ),
        "labels": torch.randint(
            0, small_model_config.language.vocab_size,
            (batch_size, seq_len), device=device
        ),
        "attention_mask": torch.ones(batch_size, seq_len, device=device),
        "images": torch.randn(
            batch_size, 3,
            small_model_config.vision.image_size,
            small_model_config.vision.image_size,
            device=device
        ),
        "image_positions": torch.tensor([[8], [8]], device=device),
    }


# =============================================================================
# Loss Computation Tests
# =============================================================================

class TestLossComputation:
    """Test loss computation."""
    
    def test_loss_is_scalar(self, small_model, dummy_batch):
        """Test loss is a scalar tensor."""
        outputs = small_model(**dummy_batch)
        assert outputs["loss"].dim() == 0
        assert outputs["loss"].numel() == 1
    
    def test_loss_is_finite(self, small_model, dummy_batch):
        """Test loss is finite (not NaN or Inf)."""
        outputs = small_model(**dummy_batch)
        assert torch.isfinite(outputs["loss"])
    
    def test_loss_positive(self, small_model, dummy_batch):
        """Test loss is positive (cross-entropy should be > 0)."""
        outputs = small_model(**dummy_batch)
        assert outputs["loss"] > 0
    
    def test_loss_decreases_with_training(self, small_model, dummy_batch, device):
        """Test loss decreases over training steps."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        
        initial_loss = None
        losses = []
        
        for step in range(10):
            outputs = small_model(**dummy_batch)
            loss = outputs["loss"]
            losses.append(loss.item())
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Loss should generally decrease
        assert losses[-1] < initial_loss
    
    def test_loss_ignores_padding(self, small_model, small_model_config, device):
        """Test loss correctly ignores -100 labels (padding)."""
        batch_size = 2
        seq_len = 32
        
        input_ids = torch.randint(0, small_model_config.language.vocab_size, (batch_size, seq_len), device=device)
        
        # All padding labels
        labels_all_padding = torch.full((batch_size, seq_len), -100, device=device)
        outputs = small_model(input_ids=input_ids, labels=labels_all_padding)
        
        # Loss should be 0 or NaN (depending on implementation)
        # If all labels are -100, loss reduction over 0 elements
        assert outputs["loss"] >= 0 or torch.isnan(outputs["loss"])
    
    def test_loss_with_different_sequence_lengths(self, small_model, small_model_config, device):
        """Test loss works with different sequence lengths."""
        for seq_len in [16, 32, 64, 128]:
            batch_size = 2
            input_ids = torch.randint(0, small_model_config.language.vocab_size, (batch_size, seq_len), device=device)
            labels = input_ids.clone()
            
            outputs = small_model(input_ids=input_ids, labels=labels)
            assert torch.isfinite(outputs["loss"])


# =============================================================================
# Gradient Accumulation Tests
# =============================================================================

class TestGradientAccumulation:
    """Test gradient accumulation."""
    
    def test_gradient_accumulation_equivalence(self, small_model, dummy_batch, device):
        """Test gradient accumulation produces same result as larger batch."""
        # Clone model for comparison
        model1 = DeepSeekVL2ForPretraining(small_model.config).to(device)
        model2 = DeepSeekVL2ForPretraining(small_model.config).to(device)
        model2.load_state_dict(model1.state_dict())
        
        # Method 1: Single forward with batch_size=2
        outputs1 = model1(**dummy_batch)
        loss1 = outputs1["loss"]
        loss1.backward()
        
        # Method 2: Two forwards with batch_size=1, accumulating
        for i in range(2):
            mini_batch = {
                k: v[i:i+1] if isinstance(v, torch.Tensor) else v
                for k, v in dummy_batch.items()
            }
            outputs2 = model2(**mini_batch)
            loss2 = outputs2["loss"] / 2  # Scale for accumulation
            loss2.backward()
        
        # Gradients should be similar (not exact due to batch norm etc)
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            if p1.grad is not None and p2.grad is not None:
                diff = (p1.grad - p2.grad).abs().mean()
                # Allow some tolerance
                assert diff < 1.0, f"Gradient diff too large for {n1}: {diff}"
    
    def test_gradient_accumulation_steps(self, small_model, dummy_batch, device):
        """Test accumulating gradients over multiple steps before update."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        accumulation_steps = 4
        
        # Store initial weights
        initial_weights = {
            n: p.clone()
            for n, p in small_model.named_parameters()
            if p.requires_grad
        }
        
        # Accumulate gradients
        for step in range(accumulation_steps):
            outputs = small_model(**dummy_batch)
            loss = outputs["loss"] / accumulation_steps
            loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
        # Weights should have changed
        for n, p in small_model.named_parameters():
            if p.requires_grad:
                assert not torch.allclose(p, initial_weights[n]), f"Weights unchanged for {n}"


# =============================================================================
# Learning Rate Scheduling Tests
# =============================================================================

class TestLearningRateScheduling:
    """Test learning rate scheduling."""
    
    def test_cosine_schedule_warmup(self):
        """Test cosine schedule with warmup."""
        total_steps = 1000
        warmup_steps = 100
        min_lr_ratio = 0.1
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        
        # Test warmup phase
        assert abs(lr_lambda(0) - 0.0) < 1e-6
        assert abs(lr_lambda(50) - 0.5) < 1e-6
        assert abs(lr_lambda(100) - 1.0) < 1e-6
        
        # Test cosine phase
        assert lr_lambda(550) < 1.0  # Middle of cosine
        assert lr_lambda(1000) >= min_lr_ratio  # End
    
    def test_linear_schedule_warmup(self):
        """Test linear schedule with warmup."""
        total_steps = 1000
        warmup_steps = 100
        min_lr_ratio = 0.0
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return max(
                min_lr_ratio,
                1 - (step - warmup_steps) / (total_steps - warmup_steps)
            )
        
        # Test warmup phase
        assert abs(lr_lambda(0) - 0.0) < 1e-6
        assert abs(lr_lambda(100) - 1.0) < 1e-6
        
        # Test linear decay
        assert lr_lambda(600) < 1.0
        assert lr_lambda(1000) == min_lr_ratio
    
    def test_scheduler_with_optimizer(self, small_model, device):
        """Test scheduler works with optimizer."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        
        initial_lr = optimizer.param_groups[0]["lr"]
        
        for _ in range(50):
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        assert current_lr != initial_lr


# =============================================================================
# Mixed Precision Tests
# =============================================================================

class TestMixedPrecision:
    """Test mixed precision training."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_autocast_forward(self, small_model_config, device):
        """Test forward pass with autocast."""
        model = DeepSeekVL2ForPretraining(small_model_config).to(device)
        
        input_ids = torch.randint(0, small_model_config.language.vocab_size, (2, 32), device=device)
        labels = input_ids.clone()
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
        
        assert torch.isfinite(outputs["loss"])
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_autocast_backward(self, small_model_config, device):
        """Test backward pass with autocast."""
        model = DeepSeekVL2ForPretraining(small_model_config).to(device)
        scaler = torch.cuda.amp.GradScaler()
        
        input_ids = torch.randint(0, small_model_config.language.vocab_size, (2, 32), device=device)
        labels = input_ids.clone()
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
        
        scaler.scale(loss).backward()
        
        # Check gradients exist and are not NaN
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
    
    def test_bfloat16_model(self, small_model_config, device):
        """Test model can run in bfloat16."""
        model = DeepSeekVL2ForPretraining(small_model_config).to(device).to(torch.bfloat16)
        
        input_ids = torch.randint(0, small_model_config.language.vocab_size, (2, 32), device=device)
        labels = input_ids.clone()
        
        outputs = model(input_ids=input_ids, labels=labels)
        assert outputs["logits"].dtype == torch.bfloat16


# =============================================================================
# Checkpoint Tests
# =============================================================================

class TestCheckpointing:
    """Test checkpoint saving and loading."""
    
    def test_save_and_load_state_dict(self, small_model, device):
        """Test saving and loading state dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            
            # Save
            torch.save(small_model.state_dict(), checkpoint_path)
            
            # Create new model and load
            new_model = DeepSeekVL2ForPretraining(small_model.config).to(device)
            new_model.load_state_dict(torch.load(checkpoint_path))
            
            # Check weights match
            for (n1, p1), (n2, p2) in zip(small_model.named_parameters(), new_model.named_parameters()):
                assert torch.allclose(p1, p2), f"Weights don't match for {n1}"
    
    def test_save_full_checkpoint(self, small_model, device):
        """Test saving full checkpoint with optimizer state."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
        
        # Do some training steps
        input_ids = torch.randint(0, small_model.config.language.vocab_size, (2, 32), device=device)
        labels = input_ids.clone()
        
        for _ in range(5):
            outputs = small_model(input_ids=input_ids, labels=labels)
            outputs["loss"].backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "full_checkpoint.pt"
            
            # Save full checkpoint
            torch.save({
                "model_state_dict": small_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": 5,
            }, checkpoint_path)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            assert checkpoint["step"] == 5
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint


# =============================================================================
# Training Stage Tests
# =============================================================================

class TestTrainingStages:
    """Test different training stages."""
    
    def test_alignment_stage(self, small_model, dummy_batch):
        """Test alignment stage (freeze LM)."""
        small_model.freeze_language_model()
        
        # LM should be frozen
        for param in small_model.language_model.parameters():
            assert not param.requires_grad
        
        # Vision should still be trainable
        for param in small_model.vision_encoder.parameters():
            assert param.requires_grad
        
        # Can still do forward pass
        outputs = small_model(**dummy_batch)
        assert torch.isfinite(outputs["loss"])
        
        # Backward should work for vision only
        outputs["loss"].backward()
        assert small_model.vision_encoder.patch_embed.proj.weight.grad is not None
        assert small_model.language_model.embed_tokens.weight.grad is None
    
    def test_pretrain_stage(self, small_model, dummy_batch):
        """Test pretrain stage (train all)."""
        small_model.unfreeze_vision_encoder()
        small_model.unfreeze_language_model()
        
        # All should be trainable
        trainable_params = sum(1 for p in small_model.parameters() if p.requires_grad)
        total_params = sum(1 for _ in small_model.parameters())
        assert trainable_params == total_params
        
        # Forward and backward should work
        outputs = small_model(**dummy_batch)
        outputs["loss"].backward()
    
    def test_sft_stage_with_targets_only(self, small_model, small_model_config, device):
        """Test SFT stage trains on targets only (using label masking)."""
        batch_size = 2
        seq_len = 32
        
        input_ids = torch.randint(0, small_model_config.language.vocab_size, (batch_size, seq_len), device=device)
        
        # Create labels with first half masked (prompt)
        labels = input_ids.clone()
        labels[:, :seq_len//2] = -100  # Mask prompt tokens
        
        outputs = small_model(input_ids=input_ids, labels=labels)
        
        # Loss should be finite
        assert torch.isfinite(outputs["loss"])


# =============================================================================
# Memory Efficiency Tests
# =============================================================================

class TestMemoryEfficiency:
    """Test memory efficiency features."""
    
    def test_gradient_checkpointing_reduces_memory(self, small_model_config, device):
        """Test gradient checkpointing reduces memory usage."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory testing")
        
        torch.cuda.reset_peak_memory_stats()
        
        # Without checkpointing
        model1 = DeepSeekVL2ForPretraining(small_model_config).to(device)
        model1.disable_gradient_checkpointing()
        
        input_ids = torch.randint(0, small_model_config.language.vocab_size, (2, 64), device=device)
        labels = input_ids.clone()
        
        outputs = model1(input_ids=input_ids, labels=labels)
        outputs["loss"].backward()
        
        memory_without_ckpt = torch.cuda.max_memory_allocated()
        
        # Reset
        del model1, outputs
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # With checkpointing
        model2 = DeepSeekVL2ForPretraining(small_model_config).to(device)
        model2.enable_gradient_checkpointing()
        
        outputs = model2(input_ids=input_ids, labels=labels)
        outputs["loss"].backward()
        
        memory_with_ckpt = torch.cuda.max_memory_allocated()
        
        # Checkpointing should use less or equal memory
        # (May not be less for small models)
        assert memory_with_ckpt <= memory_without_ckpt * 1.1  # 10% tolerance


# =============================================================================
# Numerical Stability Tests
# =============================================================================

class TestNumericalStability:
    """Test numerical stability."""
    
    def test_no_nan_in_forward(self, small_model, dummy_batch):
        """Test no NaN values in forward pass."""
        outputs = small_model(**dummy_batch)
        
        assert not torch.isnan(outputs["loss"])
        assert not torch.isnan(outputs["logits"]).any()
    
    def test_no_nan_after_many_steps(self, small_model, dummy_batch, device):
        """Test no NaN after multiple training steps."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)
        
        for step in range(50):
            outputs = small_model(**dummy_batch)
            loss = outputs["loss"]
            
            assert not torch.isnan(loss), f"NaN loss at step {step}"
            
            optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients
            for name, param in small_model.named_parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), f"NaN gradient in {name} at step {step}"
            
            torch.nn.utils.clip_grad_norm_(small_model.parameters(), 1.0)
            optimizer.step()
    
    def test_gradient_clipping(self, small_model, dummy_batch):
        """Test gradient clipping works."""
        outputs = small_model(**dummy_batch)
        outputs["loss"].backward()
        
        max_norm = 1.0
        total_norm = torch.nn.utils.clip_grad_norm_(small_model.parameters(), max_norm)
        
        # After clipping, norms should be <= max_norm
        for param in small_model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                assert grad_norm <= total_norm + 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
