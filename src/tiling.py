"""
Dynamic Tiling and M-RoPE for PixelLM.

Industry-standard features:
  - Dynamic Tiling (DeepSeek VL2 / Qwen2-VL style)
  - M-RoPE: Multimodal Rotary Position Embedding (Qwen2-VL style)
  - AnyRes: Any resolution image processing
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Dynamic Tiling (DeepSeek VL2 Style)
# =============================================================================

@dataclass
class TileConfig:
    """Configuration for dynamic tiling."""
    tile_size: int = 384
    min_tiles: int = 1
    max_tiles: int = 12
    use_thumbnail: bool = True


def get_optimal_tiling(
    image_width: int,
    image_height: int,
    tile_size: int = 384,
    max_tiles: int = 12,
) -> Tuple[int, int]:
    """
    Calculate optimal number of tiles for an image.
    
    Returns (tiles_x, tiles_y) that minimizes aspect ratio distortion.
    """
    aspect_ratio = image_width / image_height
    
    best_tiles = (1, 1)
    best_score = float('inf')
    
    for total in range(1, max_tiles + 1):
        for tx in range(1, total + 1):
            ty = total // tx
            if tx * ty > max_tiles or ty == 0:
                continue
            
            tile_aspect = tx / ty
            # Score: difference from original aspect ratio
            score = abs(tile_aspect - aspect_ratio)
            
            if score < best_score:
                best_score = score
                best_tiles = (tx, ty)
    
    return best_tiles


def tile_image(
    image: torch.Tensor,
    tile_size: int = 384,
    max_tiles: int = 12,
    use_thumbnail: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tile image into patches + optional thumbnail.
    
    Args:
        image: (B, C, H, W) tensor
        tile_size: Size of each tile
        max_tiles: Maximum number of tiles
        use_thumbnail: Include global thumbnail
    
    Returns:
        tiles: (B, N, C, tile_size, tile_size) where N = num_tiles + thumbnail
        tile_positions: (B, N, 2) grid positions for each tile
    """
    B, C, H, W = image.shape
    
    # Get optimal tiling
    tiles_x, tiles_y = get_optimal_tiling(W, H, tile_size, max_tiles)
    
    # Resize to fit tiles exactly
    new_w = tiles_x * tile_size
    new_h = tiles_y * tile_size
    resized = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    # Extract tiles
    tiles = resized.unfold(2, tile_size, tile_size).unfold(3, tile_size, tile_size)
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C, tile_size, tile_size)
    
    num_tiles = tiles.shape[1]
    
    # Create position grid
    positions = torch.zeros(B, num_tiles, 2, device=image.device, dtype=torch.long)
    for i in range(tiles_y):
        for j in range(tiles_x):
            idx = i * tiles_x + j
            positions[:, idx, 0] = i  # y position
            positions[:, idx, 1] = j  # x position
    
    # Add thumbnail
    if use_thumbnail:
        thumbnail = F.interpolate(image, size=(tile_size, tile_size), mode='bilinear', align_corners=False)
        thumbnail = thumbnail.unsqueeze(1)  # (B, 1, C, H, W)
        tiles = torch.cat([thumbnail, tiles], dim=1)
        
        # Thumbnail position is (-1, -1) to indicate global view
        thumb_pos = torch.full((B, 1, 2), -1, device=image.device, dtype=torch.long)
        positions = torch.cat([thumb_pos, positions], dim=1)
    
    return tiles, positions


class DynamicTileProcessor:
    """
    Process images with dynamic tiling.
    
    Supports any image resolution by tiling into fixed-size patches.
    """
    
    def __init__(self, config: TileConfig = None):
        self.config = config or TileConfig()
    
    def __call__(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process batch of images.
        
        Args:
            images: (B, C, H, W) batch of images
        
        Returns:
            tiles: (B, N, C, tile_size, tile_size)
            positions: (B, N, 2) tile grid positions
        """
        return tile_image(
            images,
            tile_size=self.config.tile_size,
            max_tiles=self.config.max_tiles,
            use_thumbnail=self.config.use_thumbnail,
        )


# =============================================================================
# M-RoPE: Multimodal Rotary Position Embedding (Qwen2-VL Style)
# =============================================================================

class MultimodalRoPE(nn.Module):
    """
    Multimodal Rotary Position Embedding.
    
    Decomposes position embedding into:
      - 1D: Text positions
      - 2D: Image positions (row, col)
      - 3D: Video positions (frame, row, col)
    
    Based on Qwen2-VL's M-RoPE.
    """
    
    def __init__(
        self,
        dim: int,
        max_position: int = 8192,
        base: float = 10000.0,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.spatial_merge_size = spatial_merge_size
        
        # Split dim for different position types
        # Text: 1D, Image: 2D (temporal + spatial)
        self.temporal_dim = dim // 4
        self.spatial_dim = dim - self.temporal_dim
        
        # Precompute inverse frequencies
        inv_freq_temporal = 1.0 / (base ** (torch.arange(0, self.temporal_dim, 2).float() / self.temporal_dim))
        inv_freq_spatial = 1.0 / (base ** (torch.arange(0, self.spatial_dim // 2, 2).float() / (self.spatial_dim // 2)))
        
        self.register_buffer("inv_freq_temporal", inv_freq_temporal, persistent=False)
        self.register_buffer("inv_freq_spatial", inv_freq_spatial, persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        image_grid_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute M-RoPE cos/sin embeddings.
        
        Args:
            x: Input tensor for shape reference
            position_ids: 1D text positions (B, L)
            image_grid_positions: 2D image positions (B, N, 2) with (row, col)
        
        Returns:
            cos, sin embeddings
        """
        B, L = x.shape[:2]
        device = x.device
        
        if position_ids is None:
            position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        
        # Temporal/text frequencies
        temporal_freqs = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq_temporal)
        temporal_emb = torch.cat([temporal_freqs, temporal_freqs], dim=-1)
        
        # Spatial frequencies (for images)
        if image_grid_positions is not None:
            # Row and column positions
            row_pos = image_grid_positions[..., 0].float()  # (B, N)
            col_pos = image_grid_positions[..., 1].float()  # (B, N)
            
            row_freqs = torch.einsum("bi,j->bij", row_pos, self.inv_freq_spatial)
            col_freqs = torch.einsum("bi,j->bij", col_pos, self.inv_freq_spatial)
            
            spatial_emb = torch.cat([row_freqs, row_freqs, col_freqs, col_freqs], dim=-1)
        else:
            # Default: use 1D positions for spatial too
            spatial_freqs = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq_spatial)
            spatial_emb = torch.cat([spatial_freqs, spatial_freqs], dim=-1)
        
        # Combine temporal and spatial
        emb = torch.cat([temporal_emb, spatial_emb[..., :self.spatial_dim]], dim=-1)
        
        return emb.cos().unsqueeze(2), emb.sin().unsqueeze(2)


def apply_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply multimodal rotary position embedding."""
    
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin
    
    return q_rotated, k_rotated


# =============================================================================
# 2D RoPE for Vision Transformers
# =============================================================================

class RoPE2D(nn.Module):
    """
    2D Rotary Position Embedding for Vision Transformers.
    
    Applies separate RoPE for height and width dimensions.
    """
    
    def __init__(self, dim: int, max_h: int = 64, max_w: int = 64, base: float = 10000.0):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D RoPE"
        
        self.dim = dim
        half_dim = dim // 4
        
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute positions
        h_pos = torch.arange(max_h)
        w_pos = torch.arange(max_w)
        
        h_freqs = torch.einsum("i,j->ij", h_pos.float(), inv_freq)
        w_freqs = torch.einsum("i,j->ij", w_pos.float(), inv_freq)
        
        self.register_buffer("h_freqs", h_freqs, persistent=False)
        self.register_buffer("w_freqs", w_freqs, persistent=False)
    
    def forward(self, x: torch.Tensor, grid_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 2D RoPE for patches.
        
        Args:
            x: (B, N, D) patch embeddings
            grid_size: (H, W) grid dimensions
        
        Returns:
            cos, sin embeddings (1, N, 1, D)
        """
        H, W = grid_size
        N = H * W
        
        # Get frequencies
        h_freqs = self.h_freqs[:H]  # (H, half_dim//2)
        w_freqs = self.w_freqs[:W]  # (W, half_dim//2)
        
        # Create 2D grid
        h_emb = h_freqs.unsqueeze(1).expand(-1, W, -1)  # (H, W, half_dim//2)
        w_emb = w_freqs.unsqueeze(0).expand(H, -1, -1)  # (H, W, half_dim//2)
        
        # Combine and reshape
        emb = torch.cat([
            h_emb, h_emb,  # Height cos/sin
            w_emb, w_emb,  # Width cos/sin
        ], dim=-1).reshape(N, self.dim)
        
        return emb.cos().unsqueeze(0).unsqueeze(2), emb.sin().unsqueeze(0).unsqueeze(2)


# =============================================================================
# Tile Position Embedding
# =============================================================================

class TilePositionEmbedding(nn.Module):
    """
    Learnable position embeddings for tiles.
    
    Supports both absolute tile positions and relative positions.
    """
    
    def __init__(
        self,
        hidden_size: int,
        max_tiles_h: int = 8,
        max_tiles_w: int = 8,
        use_2d_rope: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_2d_rope = use_2d_rope
        
        # Absolute embeddings for tile positions
        self.tile_h_embed = nn.Embedding(max_tiles_h + 1, hidden_size)  # +1 for thumbnail
        self.tile_w_embed = nn.Embedding(max_tiles_w + 1, hidden_size)
        
        # Thumbnail embedding
        self.thumbnail_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        if use_2d_rope:
            self.rope_2d = RoPE2D(hidden_size)
    
    def forward(
        self,
        x: torch.Tensor,
        tile_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add tile position embeddings.
        
        Args:
            x: (B, N_tiles, N_patches, D) tile features
            tile_positions: (B, N_tiles, 2) tile positions (row, col)
        
        Returns:
            x with position embeddings added
        """
        B, N_tiles, N_patches, D = x.shape
        
        # Separate thumbnail and regular tiles
        is_thumbnail = (tile_positions[..., 0] == -1)
        
        # Regular tile embeddings
        h_pos = tile_positions[..., 0].clamp(min=0)
        w_pos = tile_positions[..., 1].clamp(min=0)
        
        h_emb = self.tile_h_embed(h_pos).unsqueeze(2)  # (B, N_tiles, 1, D)
        w_emb = self.tile_w_embed(w_pos).unsqueeze(2)
        
        tile_emb = h_emb + w_emb
        
        # Replace thumbnail positions with special embedding
        thumbnail_emb = self.thumbnail_embed.expand(B, -1, -1, -1)
        tile_emb = torch.where(
            is_thumbnail.unsqueeze(-1).unsqueeze(-1).expand_as(tile_emb),
            thumbnail_emb.expand_as(tile_emb),
            tile_emb,
        )
        
        return x + tile_emb


# Export
__all__ = [
    "TileConfig",
    "DynamicTileProcessor",
    "tile_image",
    "get_optimal_tiling",
    "MultimodalRoPE",
    "apply_mrope",
    "RoPE2D",
    "TilePositionEmbedding",
]
