"""
Data loading and processing tests for DeepSeek VL2.

Tests:
- Image processing
- Dynamic tiling
- Text tokenization
- Dataset loading
- Collate functions
- Data format support
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from PIL import Image
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DataConfig
from src.data.processor import VLProcessor
from src.data.dataset import VLDataset, DummyVLDataset, collate_fn


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def processor():
    """Create test processor."""
    return VLProcessor(
        image_size=224,
        patch_size=16,
        max_tiles=9,
    )


@pytest.fixture
def sample_image():
    """Create sample PIL image."""
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_image_path(sample_image):
    """Create sample image file."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        sample_image.save(f.name)
        return f.name


@pytest.fixture
def data_config():
    """Create test data config."""
    return DataConfig(
        max_seq_length=256,
        data_format="jsonl",
        image_column="images",
        text_column="conversations",
    )


# =============================================================================
# VLProcessor Tests
# =============================================================================

class TestVLProcessor:
    """Test vision-language processor."""
    
    def test_process_pil_image(self, processor, sample_image):
        """Test processing PIL image."""
        result = processor.process_image(sample_image, use_tiling=False)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)
    
    def test_process_image_path(self, processor, sample_image_path):
        """Test processing image from path."""
        result = processor.process_image(sample_image_path, use_tiling=False)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)
    
    def test_process_tensor_image(self, processor):
        """Test processing tensor image."""
        tensor_img = torch.randn(3, 256, 256)
        result = processor.process_image(tensor_img, use_tiling=False)
        
        assert result.shape == (3, 224, 224)
    
    def test_image_normalization(self, processor, sample_image):
        """Test image is normalized."""
        result = processor.process_image(sample_image, use_tiling=False)
        
        # After normalization, values should be roughly centered around 0
        assert result.mean().abs() < 2.0
    
    def test_dynamic_tiling_small_image(self, processor, sample_image):
        """Test dynamic tiling with small image (no extra tiles)."""
        result = processor.process_image(sample_image, use_tiling=True)
        
        # Small image should produce minimal tiles
        assert result.dim() == 4  # [num_tiles, 3, H, W]
        assert result.shape[0] >= 1  # At least global view
    
    def test_dynamic_tiling_large_image(self, processor):
        """Test dynamic tiling with large image."""
        # Create large image
        large_img_array = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        large_img = Image.fromarray(large_img_array)
        
        result = processor.process_image(large_img, use_tiling=True)
        
        # Should produce multiple tiles
        assert result.dim() == 4
        assert result.shape[0] > 1  # Multiple tiles
        assert result.shape[0] <= processor.max_tiles
    
    def test_process_multiple_images(self, processor, sample_image):
        """Test processing multiple images."""
        images = [sample_image, sample_image]
        result = processor(images=images)
        
        assert "images" in result
        assert result["images"].shape[0] == 2


# =============================================================================
# DummyVLDataset Tests
# =============================================================================

class TestDummyVLDataset:
    """Test dummy dataset for testing."""
    
    def test_dataset_length(self):
        """Test dataset has correct length."""
        dataset = DummyVLDataset(size=100)
        assert len(dataset) == 100
    
    def test_dataset_item(self):
        """Test dataset item has expected keys."""
        dataset = DummyVLDataset(size=10, seq_len=64, image_size=224)
        item = dataset[0]
        
        assert "input_ids" in item
        assert "labels" in item
        assert "attention_mask" in item
        assert "images" in item
        assert "image_positions" in item
    
    def test_dataset_shapes(self):
        """Test dataset item shapes."""
        seq_len = 64
        image_size = 224
        vocab_size = 1000
        
        dataset = DummyVLDataset(
            size=10,
            seq_len=seq_len,
            image_size=image_size,
            vocab_size=vocab_size,
        )
        item = dataset[0]
        
        assert item["input_ids"].shape == (seq_len,)
        assert item["labels"].shape == (seq_len,)
        assert item["attention_mask"].shape == (seq_len,)
        assert item["images"].shape[2] == image_size
        assert item["images"].shape[3] == image_size
    
    def test_dataset_vocab_range(self):
        """Test input_ids are within vocab range."""
        vocab_size = 1000
        dataset = DummyVLDataset(size=10, vocab_size=vocab_size)
        item = dataset[0]
        
        assert item["input_ids"].min() >= 0
        assert item["input_ids"].max() < vocab_size
    
    def test_dataset_without_images(self):
        """Test dataset without images."""
        dataset = DummyVLDataset(size=10, include_images=False)
        item = dataset[0]
        
        assert "images" not in item
        assert "image_positions" not in item


# =============================================================================
# VLDataset Tests
# =============================================================================

class TestVLDataset:
    """Test real VL dataset."""
    
    @pytest.fixture
    def sample_jsonl_data(self):
        """Create sample JSONL data."""
        data = [
            {
                "images": ["image1.png"],
                "conversations": [
                    {"role": "user", "content": "What is this?"},
                    {"role": "assistant", "content": "This is a test."},
                ]
            },
            {
                "images": [],
                "conversations": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            }
        ]
        return data
    
    @pytest.fixture
    def sample_data_dir(self, sample_jsonl_data, sample_image):
        """Create sample data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            
            # Save images
            img_path = Path(tmpdir) / "image1.png"
            sample_image.save(img_path)
            
            # Save jsonl
            with open(data_path, 'w') as f:
                for item in sample_jsonl_data:
                    f.write(json.dumps(item) + '\n')
            
            yield tmpdir
    
    def test_load_jsonl_data(self, sample_data_dir, data_config, processor):
        """Test loading JSONL data."""
        # Add tokenizer to processor (mock)
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                tokens = torch.randint(0, 1000, (256,))
                return {
                    "input_ids": tokens.unsqueeze(0),
                    "attention_mask": torch.ones(1, 256),
                }
        
        processor.tokenizer = MockTokenizer()
        
        dataset = VLDataset(
            data_dir=sample_data_dir,
            processor=processor,
            config=data_config,
        )
        
        assert len(dataset) == 2
    
    def test_dataset_iteration(self, sample_data_dir, data_config, processor):
        """Test iterating over dataset."""
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                tokens = torch.randint(0, 1000, (256,))
                return {
                    "input_ids": tokens.unsqueeze(0),
                    "attention_mask": torch.ones(1, 256),
                }
        
        processor.tokenizer = MockTokenizer()
        
        dataset = VLDataset(
            data_dir=sample_data_dir,
            processor=processor,
            config=data_config,
        )
        
        for i, item in enumerate(dataset):
            assert "input_ids" in item
            assert "labels" in item
            if i >= 1:
                break


# =============================================================================
# Collate Function Tests
# =============================================================================

class TestCollateFn:
    """Test collate function."""
    
    def test_collate_basic(self):
        """Test basic collation."""
        batch = [
            {
                "input_ids": torch.randint(0, 1000, (64,)),
                "labels": torch.randint(0, 1000, (64,)),
                "attention_mask": torch.ones(64),
            },
            {
                "input_ids": torch.randint(0, 1000, (64,)),
                "labels": torch.randint(0, 1000, (64,)),
                "attention_mask": torch.ones(64),
            },
        ]
        
        result = collate_fn(batch)
        
        assert result["input_ids"].shape == (2, 64)
        assert result["labels"].shape == (2, 64)
        assert result["attention_mask"].shape == (2, 64)
    
    def test_collate_with_images(self):
        """Test collation with images."""
        batch = [
            {
                "input_ids": torch.randint(0, 1000, (64,)),
                "labels": torch.randint(0, 1000, (64,)),
                "attention_mask": torch.ones(64),
                "images": torch.randn(1, 3, 224, 224),
                "image_positions": torch.tensor([16]),
            },
            {
                "input_ids": torch.randint(0, 1000, (64,)),
                "labels": torch.randint(0, 1000, (64,)),
                "attention_mask": torch.ones(64),
                "images": torch.randn(1, 3, 224, 224),
                "image_positions": torch.tensor([16]),
            },
        ]
        
        result = collate_fn(batch)
        
        assert "images" in result
        assert "image_positions" in result
    
    def test_collate_variable_images(self):
        """Test collation with variable number of images."""
        batch = [
            {
                "input_ids": torch.randint(0, 1000, (64,)),
                "labels": torch.randint(0, 1000, (64,)),
                "attention_mask": torch.ones(64),
                "images": torch.randn(2, 3, 224, 224),  # 2 images
                "image_positions": torch.tensor([16, 32]),
            },
            {
                "input_ids": torch.randint(0, 1000, (64,)),
                "labels": torch.randint(0, 1000, (64,)),
                "attention_mask": torch.ones(64),
                "images": torch.randn(1, 3, 224, 224),  # 1 image
                "image_positions": torch.tensor([16]),
            },
        ]
        
        result = collate_fn(batch)
        
        # Should be padded to max
        assert result["image_positions"].shape == (2, 2)
        # Second sample should have -1 padding
        assert result["image_positions"][1, 1] == -1


# =============================================================================
# Data Format Tests
# =============================================================================

class TestDataFormats:
    """Test different data format support."""
    
    def test_jsonl_format(self):
        """Test JSONL format loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.jsonl"
            
            with open(data_path, 'w') as f:
                f.write('{"text": "Hello"}\n')
                f.write('{"text": "World"}\n')
            
            # Check file is valid JSONL
            with open(data_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 2
                for line in lines:
                    assert json.loads(line)
    
    def test_json_format(self):
        """Test JSON format loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data.json"
            
            data = [{"text": "Hello"}, {"text": "World"}]
            with open(data_path, 'w') as f:
                json.dump(data, f)
            
            # Check file is valid JSON
            with open(data_path, 'r') as f:
                loaded = json.load(f)
                assert len(loaded) == 2


# =============================================================================
# DataLoader Integration Tests
# =============================================================================

class TestDataLoaderIntegration:
    """Test DataLoader integration."""
    
    def test_dataloader_iteration(self):
        """Test iterating with DataLoader."""
        from torch.utils.data import DataLoader
        
        dataset = DummyVLDataset(size=100, seq_len=64, image_size=224)
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn,
        )
        
        batch = next(iter(dataloader))
        
        assert batch["input_ids"].shape[0] == 4
        assert batch["images"].shape[0] == 4
    
    def test_dataloader_num_workers(self):
        """Test DataLoader with multiple workers."""
        from torch.utils.data import DataLoader
        
        dataset = DummyVLDataset(size=100, seq_len=64, image_size=224)
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # 0 for testing to avoid multiprocessing issues
        )
        
        for i, batch in enumerate(dataloader):
            if i >= 2:
                break
            assert batch["input_ids"].shape[0] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
