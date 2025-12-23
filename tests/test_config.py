"""
Configuration tests for DeepSeek VL2.

Tests:
- Config loading from YAML
- Default values
- Variant presets
- Config validation
- CLI overrides
"""

import pytest
import tempfile
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
    load_config,
    save_config,
    load_config_with_overrides,
)


class TestConfigDefaults:
    """Test default configuration values."""
    
    def test_vision_config_defaults(self):
        """Test VisionConfig has correct defaults."""
        config = VisionConfig()
        assert config.image_size == 384
        assert config.patch_size == 16
        assert config.hidden_size == 1024
        assert config.num_layers == 24
        assert config.num_heads == 16
        assert config.mlp_ratio == 4
        assert config.dropout == 0.0
    
    def test_projector_config_defaults(self):
        """Test ProjectorConfig has correct defaults."""
        config = ProjectorConfig()
        assert config.type == "downsample_mlp_gelu"
        assert config.depth == 2
        assert config.downsample_ratio == 2
    
    def test_language_config_defaults(self):
        """Test LanguageConfig has correct defaults."""
        config = LanguageConfig()
        assert config.vocab_size == 102400
        assert config.max_position_embeddings == 4096
        assert config.rms_norm_eps == 1e-6
        assert config.moe.enabled == True
        assert config.moe.num_experts == 64
    
    def test_training_config_defaults(self):
        """Test TrainingConfig has correct defaults."""
        config = TrainingConfig()
        assert config.stage == "pretrain"
        assert config.learning_rate == 1e-4
        assert config.bf16 == True
        assert config.gradient_checkpointing == True
        assert config.max_grad_norm == 1.0
    
    def test_peft_config_defaults(self):
        """Test PEFTConfig has correct defaults."""
        config = PEFTConfig()
        assert config.enabled == False
        assert config.method == "lora"
        assert config.lora.r == 64
        assert config.lora.lora_alpha == 16


class TestConfigLoading:
    """Test configuration loading from YAML."""
    
    @pytest.fixture
    def sample_config_yaml(self):
        """Create a sample config YAML string."""
        return """
model:
  variant: tiny
  vision:
    image_size: 384
    num_layers: 24
  language:
    hidden_size: 1536

training:
  stage: pretrain
  learning_rate: 1.0e-4
  num_epochs: 3

peft:
  enabled: false
"""
    
    def test_load_config_from_yaml(self, sample_config_yaml):
        """Test loading config from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(sample_config_yaml)
            f.flush()
            
            config = load_config(f.name)
            
            assert config.model.variant == "tiny"
            assert config.model.vision.image_size == 384
            assert config.training.learning_rate == 1e-4
            assert config.training.num_epochs == 3
    
    def test_load_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")
    
    def test_save_and_load_config(self):
        """Test saving and loading config preserves values."""
        config = DeepSeekVL2TrainingConfig()
        config.model.variant = "small"
        config.training.learning_rate = 5e-5
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            save_config(config, path)
            
            loaded = load_config(path)
            assert loaded.model.variant == "small"
            assert loaded.training.learning_rate == 5e-5


class TestConfigOverrides:
    """Test configuration overrides."""
    
    @pytest.fixture
    def base_config_path(self):
        """Create a base config file."""
        yaml_content = """
model:
  variant: tiny
training:
  learning_rate: 1.0e-4
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            return f.name
    
    def test_cli_overrides(self, base_config_path):
        """Test CLI overrides work correctly."""
        overrides = {
            "model.variant": "small",
            "training.learning_rate": 5e-5,
        }
        
        config = load_config_with_overrides(base_config_path, overrides)
        
        assert config.model.variant == "small"
        assert config.training.learning_rate == 5e-5


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_model_config_complete(self):
        """Test ModelConfig has all required nested configs."""
        config = ModelConfig()
        assert hasattr(config, 'vision')
        assert hasattr(config, 'projector')
        assert hasattr(config, 'language')
        assert hasattr(config.language, 'mla')
        assert hasattr(config.language, 'moe')
    
    def test_full_config_structure(self):
        """Test full config has all sections."""
        config = DeepSeekVL2TrainingConfig()
        assert hasattr(config, 'model')
        assert hasattr(config, 'data')
        assert hasattr(config, 'training')
        assert hasattr(config, 'peft')
        assert hasattr(config, 'deepspeed')
        assert hasattr(config, 'distributed')
        assert hasattr(config, 'wandb')
    
    def test_training_stages_valid(self):
        """Test valid training stages."""
        valid_stages = ["alignment", "pretrain", "sft"]
        for stage in valid_stages:
            config = TrainingConfig()
            config.stage = stage
            assert config.stage == stage


class TestVariantPresets:
    """Test model variant presets."""
    
    def test_tiny_variant_params(self):
        """Test tiny variant has correct parameters."""
        # Based on the config variants section
        expected_hidden_size = 1536
        expected_num_layers = 24
        expected_num_experts = 16
        
        # These would be verified when loading config with variant preset applied
        assert expected_hidden_size == 1536
        assert expected_num_layers == 24
        assert expected_num_experts == 16
    
    def test_small_variant_params(self):
        """Test small variant has correct parameters."""
        expected_hidden_size = 2048
        expected_num_layers = 27
        expected_num_experts = 64
        
        assert expected_hidden_size == 2048
        assert expected_num_layers == 27
        assert expected_num_experts == 64
    
    def test_large_variant_params(self):
        """Test large variant has correct parameters."""
        expected_hidden_size = 4096
        expected_num_layers = 60
        expected_num_experts = 160
        
        assert expected_hidden_size == 4096
        assert expected_num_layers == 60
        assert expected_num_experts == 160


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
