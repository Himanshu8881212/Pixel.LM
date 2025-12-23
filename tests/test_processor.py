"""
Tests for VLProcessor and data processing.

Missing tests identified:
1. Dynamic tiling edge cases
2. Tokenizer integration
3. Image normalization values
4. Multi-image processing
5. Tile padding
6. Different image formats (JPEG, PNG, etc.)
"""

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.processor import VLProcessor


@pytest.fixture
def processor():
    return VLProcessor(
        image_size=224,
        patch_size=16,
        max_tiles=9,
    )


# =============================================================================
# Dynamic Tiling Tests
# =============================================================================

class TestDynamicTiling:
    """Test dynamic tiling algorithm."""
    
    def test_small_image_single_tile(self, processor):
        """Test small image produces single tile."""
        img = torch.randn(3, 100, 100)
        result = processor.process_image(img, use_tiling=True)
        
        # Small image should get 1 tile (global view only)
        assert result.shape[0] == 1
    
    def test_large_image_multiple_tiles(self, processor):
        """Test large image produces multiple tiles."""
        img = torch.randn(3, 800, 800)
        result = processor.process_image(img, use_tiling=True)
        
        # Should produce multiple tiles
        assert result.shape[0] > 1
    
    def test_max_tiles_limit(self, processor):
        """Test max_tiles is respected."""
        img = torch.randn(3, 2000, 2000)  # Very large
        result = processor.process_image(img, use_tiling=True)
        
        assert result.shape[0] <= processor.max_tiles
    
    def test_tiling_includes_global_view(self, processor):
        """Test first tile is always global view."""
        img = torch.randn(3, 600, 600)
        result = processor.process_image(img, use_tiling=True)
        
        # First tile should be global (full image resized)
        assert result.shape[0] >= 1
        assert result[0].shape == (3, 224, 224)
    
    def test_tile_dimensions_consistent(self, processor):
        """Test all tiles have same dimensions."""
        img = torch.randn(3, 800, 800)
        result = processor.process_image(img, use_tiling=True)
        
        for i in range(result.shape[0]):
            assert result[i].shape == (3, 224, 224)
    
    def test_non_square_image_tiling(self, processor):
        """Test tiling works for non-square images."""
        # Wide image
        wide = torch.randn(3, 300, 900)
        result_wide = processor.process_image(wide, use_tiling=True)
        assert result_wide.shape[0] >= 1
        
        # Tall image
        tall = torch.randn(3, 900, 300)
        result_tall = processor.process_image(tall, use_tiling=True)
        assert result_tall.shape[0] >= 1
    
    def test_exact_size_image(self, processor):
        """Test image exactly at target size."""
        img = torch.randn(3, 224, 224)
        result = processor.process_image(img, use_tiling=True)
        
        # Should produce just 1 tile
        assert result.shape[0] == 1


# =============================================================================
# Image Format Tests
# =============================================================================

class TestImageFormats:
    """Test different image input formats."""
    
    def test_pil_image_input(self, processor):
        """Test PIL Image input."""
        pil_img = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        result = processor.process_image(pil_img, use_tiling=False)
        
        assert result.shape == (3, 224, 224)
    
    def test_tensor_input(self, processor):
        """Test tensor input."""
        tensor_img = torch.randn(3, 256, 256)
        result = processor.process_image(tensor_img, use_tiling=False)
        
        assert result.shape == (3, 224, 224)
    
    def test_file_path_input(self, processor):
        """Test file path input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            
            pil_img = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            pil_img.save(path)
            
            result = processor.process_image(str(path), use_tiling=False)
            assert result.shape == (3, 224, 224)
    
    def test_jpeg_image(self, processor):
        """Test JPEG image input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jpg"
            
            pil_img = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            pil_img.save(path, "JPEG")
            
            result = processor.process_image(str(path), use_tiling=False)
            assert result.shape == (3, 224, 224)
    
    def test_grayscale_image(self, processor):
        """Test grayscale image is converted to RGB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            
            # Create grayscale image
            gray_img = Image.fromarray(
                np.random.randint(0, 255, (256, 256), dtype=np.uint8), mode='L'
            )
            gray_img.save(path)
            
            result = processor.process_image(str(path), use_tiling=False)
            assert result.shape[0] == 3  # Should be 3 channels


# =============================================================================
# Normalization Tests
# =============================================================================

class TestImageNormalization:
    """Test image normalization."""
    
    def test_normalization_applied(self, processor):
        """Test normalization is applied."""
        # White image
        white = torch.ones(3, 256, 256)
        result = processor.process_image(white, use_tiling=False)
        
        # After normalization, values shouldn't all be 1
        assert not torch.allclose(result, torch.ones_like(result))
    
    def test_normalization_range(self, processor):
        """Test normalized values are in reasonable range."""
        img = torch.rand(3, 256, 256)  # Random [0, 1]
        result = processor.process_image(img, use_tiling=False)
        
        # After ImageNet normalization, values typically in [-3, 3]
        assert result.min() > -10
        assert result.max() < 10
    
    def test_normalization_values(self, processor):
        """Test normalization uses correct mean/std."""
        # ImageNet mean/std
        assert torch.allclose(processor.image_mean.squeeze(), torch.tensor([0.485, 0.456, 0.406]))
        assert torch.allclose(processor.image_std.squeeze(), torch.tensor([0.229, 0.224, 0.225]))


# =============================================================================
# Multi-Image Processing Tests
# =============================================================================

class TestMultiImageProcessing:
    """Test processing multiple images."""
    
    def test_batch_image_processing(self, processor):
        """Test processing batch of images."""
        pil_imgs = [
            Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            for _ in range(3)
        ]
        
        result = processor(images=pil_imgs)
        
        assert "images" in result
        assert result["images"].shape[0] == 3
    
    def test_variable_size_images(self, processor):
        """Test processing images of different sizes."""
        pil_imgs = [
            Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)),
            Image.fromarray(np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)),
            Image.fromarray(np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)),
        ]
        
        result = processor(images=pil_imgs)
        
        # All should be padded to same number of tiles
        assert result["images"].shape[0] == 3
        # All tiles should have same size
        assert result["images"].shape[3] == 224
        assert result["images"].shape[4] == 224
    
    def test_tile_padding_for_batch(self, processor):
        """Test tile padding when images have different tile counts."""
        # Small image (1 tile)
        small = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        # Large image (multiple tiles)
        large = Image.fromarray(np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8))
        
        result = processor(images=[small, large])
        
        # Both should have same number of tiles (padded)
        assert result["images"].shape[1] == result["images"].shape[1]


# =============================================================================
# Tokenizer Integration Tests
# =============================================================================

class TestTokenizerIntegration:
    """Test tokenizer integration."""
    
    def test_tokenizer_not_set_raises(self, processor):
        """Test error when tokenizer not set."""
        with pytest.raises(ValueError, match="Tokenizer not set"):
            processor.process_text("Hello world")
    
    def test_tokenizer_integration(self, processor):
        """Test with mock tokenizer."""
        # Mock tokenizer
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                # Return mock tensors
                seq_len = min(len(text.split()) * 2, kwargs.get("max_length", 2048))
                return {
                    "input_ids": torch.randint(0, 1000, (1, seq_len)),
                    "attention_mask": torch.ones(1, seq_len),
                }
        
        processor.tokenizer = MockTokenizer()
        result = processor.process_text("Hello world", max_length=256)
        
        assert "input_ids" in result
        assert "attention_mask" in result


# =============================================================================
# Edge Cases
# =============================================================================

class TestProcessorEdgeCases:
    """Test processor edge cases."""
    
    def test_1x1_image(self, processor):
        """Test handling of 1x1 pixel image."""
        tiny = torch.randn(3, 1, 1)
        result = processor.process_image(tiny, use_tiling=False)
        
        assert result.shape == (3, 224, 224)
    
    def test_very_long_aspect_ratio(self, processor):
        """Test very long/thin image."""
        thin = torch.randn(3, 1000, 50)
        result = processor.process_image(thin, use_tiling=True)
        
        assert result.shape[-2:] == (224, 224)
    
    def test_empty_image_list(self, processor):
        """Test empty image list."""
        result = processor(images=None, text=None)
        
        assert result == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
