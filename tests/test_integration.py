"""
Integration tests and end-to-end tests for DeepSeek VL2.

Tests:
- Full training pipeline
- Model loading from config
- Inference pipeline
- Export and import
"""

import pytest
import torch
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
    MLAConfig,
    MoEConfig,
    load_config,
    save_config,
)
from src.models.deepseek_vl2 import DeepSeekVL2ForPretraining
from src.data.dataset import DummyVLDataset, collate_fn


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    """Get test device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tiny_config():
    """Create tiny model config for fast testing."""
    vision = VisionConfig(
        image_size=56,
        patch_size=8,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
    )
    projector = ProjectorConfig(
        type="downsample_mlp_gelu",
        input_dim=64,
        output_dim=128,
        depth=1,
        downsample_ratio=2,
    )
    language = LanguageConfig(
        hidden_size=128,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        head_dim=64,
        intermediate_size=256,
        vocab_size=256,
        max_position_embeddings=128,
        mla=MLAConfig(
            q_lora_rank=64,
            kv_lora_rank=32,
            qk_rope_head_dim=8,
            qk_nope_head_dim=8,
            v_head_dim=32,
        ),
        moe=MoEConfig(
            enabled=False,  # Disable MoE for speed
            num_experts=2,
            num_experts_per_token=1,
        ),
    )
    
    return ModelConfig(
        variant="tiny",
        vision=vision,
        projector=projector,
        language=language,
    )


# =============================================================================
# Integration Tests
# =============================================================================

class TestEndToEndTraining:
    """Test full training pipeline."""
    
    def test_single_training_step(self, tiny_config, device):
        """Test a single training step."""
        model = DeepSeekVL2ForPretraining(tiny_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create batch
        input_ids = torch.randint(0, tiny_config.language.vocab_size, (2, 16), device=device)
        labels = input_ids.clone()
        
        # Forward
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert torch.isfinite(loss)
    
    def test_multi_step_training(self, tiny_config, device):
        """Test multiple training steps."""
        model = DeepSeekVL2ForPretraining(tiny_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        losses = []
        for step in range(10):
            input_ids = torch.randint(0, tiny_config.language.vocab_size, (2, 16), device=device)
            labels = input_ids.clone()
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # All losses should be finite
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)
        # Loss should generally decrease
        assert losses[-1] < losses[0]
    
    def test_training_with_images(self, tiny_config, device):
        """Test training with image inputs."""
        model = DeepSeekVL2ForPretraining(tiny_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        input_ids = torch.randint(0, tiny_config.language.vocab_size, (2, 32), device=device)
        images = torch.randn(2, 3, tiny_config.vision.image_size, tiny_config.vision.image_size, device=device)
        image_positions = torch.tensor([[8], [8]], device=device)
        labels = input_ids.clone()
        
        outputs = model(
            input_ids=input_ids,
            images=images,
            image_positions=image_positions,
            labels=labels,
        )
        
        optimizer.zero_grad()
        outputs["loss"].backward()
        optimizer.step()
        
        assert torch.isfinite(outputs["loss"])
    
    def test_dataloader_training(self, tiny_config, device):
        """Test training with DataLoader."""
        from torch.utils.data import DataLoader
        
        model = DeepSeekVL2ForPretraining(tiny_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        dataset = DummyVLDataset(
            size=20,
            seq_len=32,
            image_size=tiny_config.vision.image_size,
            vocab_size=tiny_config.language.vocab_size,
        )
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch["input_ids"],
                images=batch.get("images"),
                image_positions=batch.get("image_positions"),
                labels=batch["labels"],
            )
            
            optimizer.zero_grad()
            outputs["loss"].backward()
            optimizer.step()
        
        # Training completed without errors


class TestModelSaveLoad:
    """Test model saving and loading."""
    
    def test_save_load_consistency(self, tiny_config, device):
        """Test model produces same output after save/load."""
        model = DeepSeekVL2ForPretraining(tiny_config).to(device)
        model.eval()
        
        # Create input
        torch.manual_seed(42)
        input_ids = torch.randint(0, tiny_config.language.vocab_size, (1, 16), device=device)
        
        # Get original output
        with torch.no_grad():
            original_output = model(input_ids=input_ids)
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), checkpoint_path)
            
            # Create new model and load
            new_model = DeepSeekVL2ForPretraining(tiny_config).to(device)
            new_model.load_state_dict(torch.load(checkpoint_path))
            new_model.eval()
            
            # Get loaded output
            with torch.no_grad():
                loaded_output = new_model(input_ids=input_ids)
        
        # Outputs should match
        assert torch.allclose(
            original_output["logits"],
            loaded_output["logits"],
            atol=1e-5,
        )
    
    def test_config_save_load(self):
        """Test config save/load consistency."""
        config = DeepSeekVL2TrainingConfig()
        config.model.variant = "small"
        config.training.learning_rate = 5e-5
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            save_config(config, config_path)
            loaded_config = load_config(config_path)
        
        assert loaded_config.model.variant == "small"
        assert loaded_config.training.learning_rate == 5e-5


class TestInference:
    """Test inference pipeline."""
    
    def test_text_generation(self, tiny_config, device):
        """Test basic text generation."""
        model = DeepSeekVL2ForPretraining(tiny_config).to(device)
        model.eval()
        
        input_ids = torch.randint(0, tiny_config.language.vocab_size, (1, 8), device=device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        # Get next token prediction
        logits = outputs["logits"]
        next_token_logits = logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1)
        
        assert next_token.shape == (1,)
        assert next_token.item() < tiny_config.language.vocab_size
    
    def test_batch_inference(self, tiny_config, device):
        """Test batch inference."""
        model = DeepSeekVL2ForPretraining(tiny_config).to(device)
        model.eval()
        
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            input_ids = torch.randint(0, tiny_config.language.vocab_size, (batch_size, 16), device=device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids)
            
            assert outputs["logits"].shape[0] == batch_size
    
    def test_image_understanding(self, tiny_config, device):
        """Test image understanding inference."""
        model = DeepSeekVL2ForPretraining(tiny_config).to(device)
        model.eval()
        
        input_ids = torch.randint(0, tiny_config.language.vocab_size, (1, 32), device=device)
        images = torch.randn(1, 3, tiny_config.vision.image_size, tiny_config.vision.image_size, device=device)
        image_positions = torch.tensor([[8]], device=device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                images=images,
                image_positions=image_positions,
            )
        
        assert outputs["logits"].shape == (1, 32, tiny_config.language.vocab_size)


class TestModelProperties:
    """Test model properties and behaviors."""
    
    def test_deterministic_output(self, tiny_config, device):
        """Test model produces deterministic output."""
        model = DeepSeekVL2ForPretraining(tiny_config).to(device)
        model.eval()
        
        torch.manual_seed(42)
        input_ids = torch.randint(0, tiny_config.language.vocab_size, (1, 16), device=device)
        
        with torch.no_grad():
            output1 = model(input_ids=input_ids)["logits"]
            output2 = model(input_ids=input_ids)["logits"]
        
        assert torch.allclose(output1, output2)
    
    def test_different_inputs_different_outputs(self, tiny_config, device):
        """Test different inputs produce different outputs."""
        model = DeepSeekVL2ForPretraining(tiny_config).to(device)
        model.eval()
        
        input1 = torch.randint(0, tiny_config.language.vocab_size, (1, 16), device=device)
        input2 = torch.randint(0, tiny_config.language.vocab_size, (1, 16), device=device)
        
        with torch.no_grad():
            output1 = model(input_ids=input1)["logits"]
            output2 = model(input_ids=input2)["logits"]
        
        assert not torch.allclose(output1, output2)
    
    def test_model_device_movement(self, tiny_config):
        """Test model can be moved between devices."""
        model = DeepSeekVL2ForPretraining(tiny_config)
        
        # CPU
        model = model.to("cpu")
        assert next(model.parameters()).device.type == "cpu"
        
        # GPU (if available)
        if torch.cuda.is_available():
            model = model.to("cuda")
            assert next(model.parameters()).device.type == "cuda"
    
    def test_model_dtype_conversion(self, tiny_config, device):
        """Test model can be converted to different dtypes."""
        model = DeepSeekVL2ForPretraining(tiny_config).to(device)
        
        # Float32 (default)
        assert next(model.parameters()).dtype == torch.float32
        
        # BFloat16
        model_bf16 = model.to(torch.bfloat16)
        assert next(model_bf16.parameters()).dtype == torch.bfloat16
        
        # Float16
        model_fp16 = model.to(torch.float16)
        assert next(model_fp16.parameters()).dtype == torch.float16


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_input(self, tiny_config, device):
        """Test handling of minimal input."""
        model = DeepSeekVL2ForPretraining(tiny_config).to(device)
        
        # Minimal sequence length
        input_ids = torch.randint(0, tiny_config.language.vocab_size, (1, 1), device=device)
        
        outputs = model(input_ids=input_ids)
        assert outputs["logits"].shape == (1, 1, tiny_config.language.vocab_size)
    
    def test_max_sequence_length(self, tiny_config, device):
        """Test handling of max sequence length."""
        model = DeepSeekVL2ForPretraining(tiny_config).to(device)
        
        # Max sequence length
        max_len = tiny_config.language.max_position_embeddings
        input_ids = torch.randint(0, tiny_config.language.vocab_size, (1, max_len), device=device)
        
        outputs = model(input_ids=input_ids)
        assert outputs["logits"].shape == (1, max_len, tiny_config.language.vocab_size)
    
    def test_no_images(self, tiny_config, device):
        """Test model works without images."""
        model = DeepSeekVL2ForPretraining(tiny_config).to(device)
        
        input_ids = torch.randint(0, tiny_config.language.vocab_size, (2, 16), device=device)
        
        outputs = model(input_ids=input_ids)
        assert "logits" in outputs
    
    def test_multiple_images(self, tiny_config, device):
        """Test model with multiple images per sample."""
        model = DeepSeekVL2ForPretraining(tiny_config).to(device)
        
        input_ids = torch.randint(0, tiny_config.language.vocab_size, (1, 64), device=device)
        # Multiple images
        images = torch.randn(1, 2, 3, tiny_config.vision.image_size, tiny_config.vision.image_size, device=device)
        # Note: image_positions would need to account for multiple images
        
        # This tests the model can at least process the images
        image_embeds = model.encode_images(images)
        assert image_embeds is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
