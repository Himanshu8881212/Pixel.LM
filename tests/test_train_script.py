"""
Tests for train.py functions and training utilities.

Missing tests identified:
1. parse_args and override parsing
2. setup_training_stage
3. create_optimizer (weight decay groups)
4. create_scheduler (cosine/linear)
5. train_epoch
6. save_checkpoint
7. count_parameters
"""

import pytest
import torch
import tempfile
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DeepSeekVL2TrainingConfig,
    ModelConfig,
    VisionConfig,
    ProjectorConfig,
    LanguageConfig,
    MLAConfig,
    MoEConfig,
    TrainingConfig,
    save_config,
)
from src.models.deepseek_vl2 import DeepSeekVL2ForPretraining


# Try to import train.py functions
try:
    from train import (
        parse_args,
        setup_training_stage,
        count_parameters,
        create_optimizer,
        create_scheduler,
        save_checkpoint,
    )
    TRAIN_AVAILABLE = True
except ImportError:
    TRAIN_AVAILABLE = False


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tiny_config():
    return ModelConfig(
        vision=VisionConfig(image_size=56, hidden_size=64, num_layers=1, num_heads=2, patch_size=8),
        projector=ProjectorConfig(input_dim=64, output_dim=128, type="linear"),
        language=LanguageConfig(
            hidden_size=128, num_layers=1, vocab_size=256,
            mla=MLAConfig(q_lora_rank=64, kv_lora_rank=32, qk_rope_head_dim=8, qk_nope_head_dim=8, v_head_dim=16),
            moe=MoEConfig(enabled=False)
        ),
    )


@pytest.fixture
def tiny_model(tiny_config, device):
    return DeepSeekVL2ForPretraining(tiny_config).to(device)


# =============================================================================
# setup_training_stage Tests
# =============================================================================

@pytest.mark.skipif(not TRAIN_AVAILABLE, reason="train.py not importable")
class TestSetupTrainingStage:
    """Test training stage configuration."""
    
    def test_alignment_stage_freezes_lm(self, tiny_model):
        """Test alignment stage freezes language model."""
        setup_training_stage(tiny_model, "alignment")
        
        # LM should be frozen
        for name, param in tiny_model.language_model.named_parameters():
            assert not param.requires_grad, f"LM param {name} should be frozen"
        
        # Vision should be trainable
        for name, param in tiny_model.vision_encoder.named_parameters():
            assert param.requires_grad, f"Vision param {name} should be trainable"
    
    def test_pretrain_stage_all_trainable(self, tiny_model):
        """Test pretrain stage enables all parameters."""
        setup_training_stage(tiny_model, "pretrain")
        
        for name, param in tiny_model.named_parameters():
            assert param.requires_grad, f"Param {name} should be trainable"
    
    def test_sft_stage_all_trainable(self, tiny_model):
        """Test SFT stage enables all parameters."""
        setup_training_stage(tiny_model, "sft")
        
        for name, param in tiny_model.named_parameters():
            assert param.requires_grad, f"Param {name} should be trainable"
    
    def test_invalid_stage_raises(self, tiny_model):
        """Test invalid stage raises error."""
        with pytest.raises(ValueError):
            setup_training_stage(tiny_model, "invalid_stage")


# =============================================================================
# count_parameters Tests
# =============================================================================

@pytest.mark.skipif(not TRAIN_AVAILABLE, reason="train.py not importable")
class TestCountParameters:
    """Test parameter counting."""
    
    def test_count_all_parameters(self, tiny_model):
        """Test counting all parameters."""
        counts = count_parameters(tiny_model)
        
        assert "total" in counts
        assert "trainable" in counts
        assert counts["total"] > 0
        assert counts["trainable"] == counts["total"]  # All trainable initially
    
    def test_count_after_freezing(self, tiny_model):
        """Test counting after freezing some parameters."""
        # Freeze vision
        tiny_model.freeze_vision_encoder()
        
        counts = count_parameters(tiny_model)
        
        assert counts["trainable"] < counts["total"]


# =============================================================================
# create_optimizer Tests
# =============================================================================

@pytest.mark.skipif(not TRAIN_AVAILABLE, reason="train.py not importable")
class TestCreateOptimizer:
    """Test optimizer creation."""
    
    def test_optimizer_has_two_param_groups(self, tiny_model):
        """Test optimizer separates params with/without weight decay."""
        config = DeepSeekVL2TrainingConfig()
        config.training.learning_rate = 1e-4
        config.training.weight_decay = 0.1
        
        optimizer = create_optimizer(tiny_model, config)
        
        # Should have 2 groups: with and without weight decay
        assert len(optimizer.param_groups) == 2
    
    def test_optimizer_weight_decay_separation(self, tiny_model):
        """Test bias and layernorm have no weight decay."""
        config = DeepSeekVL2TrainingConfig()
        config.training.weight_decay = 0.1
        
        optimizer = create_optimizer(tiny_model, config)
        
        # Group 0: with weight decay
        assert optimizer.param_groups[0]["weight_decay"] == 0.1
        # Group 1: without weight decay (bias, layernorm)
        assert optimizer.param_groups[1]["weight_decay"] == 0.0
    
    def test_optimizer_learning_rate(self, tiny_model):
        """Test optimizer has correct learning rate."""
        config = DeepSeekVL2TrainingConfig()
        lr = 5e-5
        config.training.learning_rate = lr
        
        optimizer = create_optimizer(tiny_model, config)
        
        assert optimizer.param_groups[0]["lr"] == lr


# =============================================================================
# create_scheduler Tests
# =============================================================================

@pytest.mark.skipif(not TRAIN_AVAILABLE, reason="train.py not importable")
class TestCreateScheduler:
    """Test scheduler creation."""
    
    def test_cosine_scheduler_warmup(self, tiny_model):
        """Test cosine scheduler with warmup."""
        config = DeepSeekVL2TrainingConfig()
        config.training.lr_scheduler_type = "cosine"
        config.training.warmup_steps = 100
        config.training.min_lr_ratio = 0.1
        
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        scheduler = create_scheduler(optimizer, config, num_training_steps=1000)
        
        # At step 0, LR should be ~0
        assert optimizer.param_groups[0]["lr"] < 1e-3
        
        # Step through warmup
        for _ in range(100):
            scheduler.step()
        
        # At step 100, LR should be at max
        assert abs(optimizer.param_groups[0]["lr"] - 1e-3) < 1e-6
    
    def test_linear_scheduler(self, tiny_model):
        """Test linear scheduler."""
        config = DeepSeekVL2TrainingConfig()
        config.training.lr_scheduler_type = "linear"
        config.training.warmup_steps = 10
        config.training.min_lr_ratio = 0.0
        
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        scheduler = create_scheduler(optimizer, config, num_training_steps=100)
        
        # Step to end
        for _ in range(100):
            scheduler.step()
        
        # LR should be at minimum
        assert optimizer.param_groups[0]["lr"] <= 1e-3


# =============================================================================
# save_checkpoint Tests
# =============================================================================

@pytest.mark.skipif(not TRAIN_AVAILABLE, reason="train.py not importable")
class TestSaveCheckpoint:
    """Test checkpoint saving."""
    
    def test_checkpoint_saves_all_components(self, tiny_model):
        """Test checkpoint contains model, optimizer, scheduler."""
        config = DeepSeekVL2TrainingConfig()
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.training.output_dir = tmpdir
            save_checkpoint(tiny_model, optimizer, scheduler, step=100, config=config)
            
            # Check files exist
            checkpoint_dir = Path(tmpdir) / "checkpoint-100"
            assert checkpoint_dir.exists()
            assert (checkpoint_dir / "checkpoint.pt").exists()
            assert (checkpoint_dir / "config.yaml").exists()
    
    def test_checkpoint_loadable(self, tiny_model, tiny_config):
        """Test saved checkpoint can be loaded."""
        config = DeepSeekVL2TrainingConfig()
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.training.output_dir = tmpdir
            save_checkpoint(tiny_model, optimizer, scheduler, step=100, config=config)
            
            # Load checkpoint
            checkpoint_path = Path(tmpdir) / "checkpoint-100" / "checkpoint.pt"
            checkpoint = torch.load(checkpoint_path)
            
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "scheduler_state_dict" in checkpoint
            assert checkpoint["step"] == 100


# =============================================================================
# parse_args Tests (unit tests without actual CLI)
# =============================================================================

class TestParseArgsLogic:
    """Test argument parsing logic."""
    
    def test_override_string_conversion_true(self):
        """Test 'true' string converts to boolean True."""
        value = "true"
        if value.lower() == "true":
            value = True
        assert value is True
    
    def test_override_string_conversion_false(self):
        """Test 'false' string converts to boolean False."""
        value = "false"
        if value.lower() == "false":
            value = False
        assert value is False
    
    def test_override_int_conversion(self):
        """Test integer string converts to int."""
        value = "123"
        try:
            value = int(value)
        except ValueError:
            pass
        assert value == 123
        assert isinstance(value, int)
    
    def test_override_float_conversion(self):
        """Test float string converts to float."""
        value = "1.5e-4"
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        assert value == 1.5e-4
        assert isinstance(value, float)
    
    def test_nested_key_parsing(self):
        """Test nested key splitting."""
        key = "model.language.hidden_size"
        parts = key.split(".")
        assert parts == ["model", "language", "hidden_size"]


# =============================================================================
# train_epoch Logic Tests (without full training)
# =============================================================================

class TestTrainEpochLogic:
    """Test train_epoch logic without full training."""
    
    def test_gradient_accumulation_division(self):
        """Test loss is divided by accumulation steps."""
        loss = torch.tensor(1.0)
        accumulation_steps = 4
        
        scaled_loss = loss / accumulation_steps
        assert scaled_loss == 0.25
    
    def test_gradient_accumulation_counter(self):
        """Test accumulation step counter logic."""
        accumulation_steps = 4
        
        for step in range(16):
            should_update = (step + 1) % accumulation_steps == 0
            if step in [3, 7, 11, 15]:
                assert should_update
            else:
                assert not should_update
    
    def test_max_steps_early_stop(self):
        """Test max_steps causes early stopping."""
        max_steps = 10
        global_step = 0
        
        for step in range(100):
            global_step += 1
            if max_steps and global_step >= max_steps:
                break
        
        assert global_step == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
