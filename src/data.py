"""
Data loading and processing for PixelLM.

Industry-standard pipeline with:
  - 70:30 VL:text mixing for Stage 2
  - Progressive resolution (224→384→512)
  - DPO dataset support for Stage 4
  - Proper dataset presets per stage
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Iterator
import json
from pathlib import Path
import io
import random

import torch
from torch.utils.data import Dataset, IterableDataset
from PIL import Image
import torch.nn.functional as F

try:
    import requests
except ImportError:
    requests = None


# =============================================================================
# Industry-Standard Dataset Presets
# =============================================================================

STAGE_DATASETS = {
    "stage1_alignment": {
        "vl_datasets": [
            "liuhaotian/LLaVA-CC3M-Pretrain-595K",
            "Lin-Chen/ShareGPT4V",
        ],
        "text_ratio": 0.0,
    },
    "stage2_pretrain": {
        "vl_datasets": [
            "laion/laion400m",
            "kakaobrain/coyo-700m", 
        ],
        "text_ratio": 0.3,  # 30% text-only (DeepSeek-VL style)
        "text_datasets": ["HuggingFaceFW/fineweb"],
    },
    "stage3_sft": {
        "vl_datasets": [
            "liuhaotian/LLaVA-Instruct-150K",
            "HuggingFaceH4/llava-instruct-mix-vsft",
        ],
        "text_ratio": 0.0,
    },
    "stage4_dpo": {
        "vl_datasets": [
            "openbmb/RLHF-V-Dataset",
        ],
        "text_ratio": 0.0,
    },
}


@dataclass
class VLDataCollator:
    """Data collator for transformers.Trainer."""
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        
        for key in ['input_ids', 'labels', 'attention_mask']:
            if key in features[0]:
                batch[key] = torch.stack([f[key] for f in features])
        
        if 'images' in features[0] and features[0]['images'] is not None:
            valid = [f for f in features if f.get('images') is not None]
            if valid:
                batch['images'] = torch.stack([f['images'] for f in valid])
        
        if 'image_positions' in features[0] and features[0]['image_positions'] is not None:
            valid = [f for f in features if f.get('image_positions') is not None]
            if valid:
                batch['image_positions'] = torch.stack([f['image_positions'] for f in valid])
        
        # DPO fields
        for key in ['chosen_input_ids', 'chosen_labels', 'rejected_input_ids', 'rejected_labels']:
            if key in features[0]:
                batch[key] = torch.stack([f[key] for f in features])
        
        return batch


# =============================================================================
# Progressive Resolution Processor
# =============================================================================

class ProgressiveProcessor:
    """
    Image processor with progressive resolution increase.
    
    Follows industry practice: start low-res, increase during training.
    """
    
    RESOLUTION_STAGES = [
        (224, 0.0),   # Start: 224px
        (384, 0.4),   # At 40%: 384px
        (512, 0.8),   # At 80%: 512px
    ]
    
    def __init__(
        self, 
        image_size: int = 384,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
        progressive: bool = False,
    ):
        self.base_size = image_size
        self.current_size = image_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.progressive = progressive
        self._progress = 0.0
    
    def set_progress(self, progress: float):
        """Update training progress (0.0 to 1.0) for resolution scheduling."""
        self._progress = progress
        if self.progressive:
            for size, threshold in reversed(self.RESOLUTION_STAGES):
                if progress >= threshold:
                    self.current_size = size
                    break
    
    @property
    def image_size(self):
        return self.current_size if self.progressive else self.base_size
    
    def __call__(self, image: Union[Image.Image, str, torch.Tensor, bytes]) -> Optional[torch.Tensor]:
        # Handle URL
        if isinstance(image, str) and image.startswith(('http://', 'https://')):
            if requests is None:
                return None
            try:
                response = requests.get(image, timeout=10)
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
            except Exception:
                return None
        elif isinstance(image, str):
            try:
                image = Image.open(image).convert('RGB')
            except Exception:
                return None
        elif isinstance(image, bytes):
            try:
                image = Image.open(io.BytesIO(image)).convert('RGB')
            except Exception:
                return None
        
        if isinstance(image, Image.Image):
            import torchvision.transforms.functional as TF
            image = TF.to_tensor(image)
        
        size = self.image_size
        image = F.interpolate(
            image.unsqueeze(0), 
            size=(size, size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        return (image - self.mean) / self.std


# Legacy alias
VLProcessor = ProgressiveProcessor


# =============================================================================
# Datasets
# =============================================================================

class DummyDataset(Dataset):
    """Dummy dataset for testing."""
    
    def __init__(self, size=1000, seq_len=512, image_size=384, vocab_size=100000, prob_image=0.5):
        self.size = size
        self.seq_len = seq_len
        self.image_size = image_size
        self.vocab_size = vocab_size
        self.prob_image = prob_image
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = input_ids.clone()
        
        sample = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.ones(self.seq_len, dtype=torch.long),
        }
        
        if torch.rand(1).item() < self.prob_image:
            sample['images'] = torch.randn(3, self.image_size, self.image_size)
            sample['image_positions'] = torch.tensor([self.seq_len // 4])
        else:
            sample['images'] = None
            sample['image_positions'] = None
        
        return sample


class TextOnlyDataset(IterableDataset):
    """
    Text-only dataset for preserving LLM capabilities.
    
    Used in 70:30 VL:text mixing during Stage 2.
    """
    
    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb",
        tokenizer = None,
        max_seq_len: int = 2048,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        from datasets import load_dataset
        self.dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for sample in self.dataset:
            try:
                text = sample.get("text", "")
                if not text or len(text) < 100:
                    continue
                
                if self.tokenizer:
                    tokens = self.tokenizer(
                        text, 
                        max_length=self.max_seq_len, 
                        truncation=True, 
                        padding="max_length", 
                        return_tensors="pt"
                    )
                    input_ids = tokens["input_ids"].squeeze(0)
                    attention_mask = tokens["attention_mask"].squeeze(0)
                else:
                    input_ids = torch.randint(0, 32000, (self.max_seq_len,))
                    attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)
                
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                    "attention_mask": attention_mask,
                    "images": None,
                    "image_positions": None,
                }
            except Exception:
                continue


class StreamingVLDataset(IterableDataset):
    """HuggingFace streaming dataset for VLM pretraining."""
    
    COLUMN_MAPPINGS = {
        "laion/laion400m": {"image": "url", "text": "caption"},
        "kakaobrain/coyo-700m": {"image": "url", "text": "text"},
        "liuhaotian/LLaVA-CC3M-Pretrain-595K": {"image": "image", "text": "conversations"},
        "Lin-Chen/ShareGPT4V": {"image": "image", "text": "conversations"},
        "google-research-datasets/conceptual_captions": {"image": "image_url", "text": "caption"},
    }
    
    def __init__(
        self,
        dataset_name: str,
        processor: ProgressiveProcessor,
        tokenizer = None,
        max_seq_len: int = 2048,
        split: str = "train",
        image_column: str = None,
        text_column: str = None,
    ):
        self.dataset_name = dataset_name
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        
        mapping = self.COLUMN_MAPPINGS.get(dataset_name, {})
        self.image_column = image_column or mapping.get("image", "image")
        self.text_column = text_column or mapping.get("text", "text")
        
        from datasets import load_dataset
        self.dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for sample in self.dataset:
            try:
                image_data = sample.get(self.image_column)
                if image_data is None:
                    continue
                
                image = self.processor(image_data)
                if image is None:
                    continue
                
                text = sample.get(self.text_column, "")
                # Handle conversation format
                if isinstance(text, list):
                    text = " ".join([t.get("value", "") for t in text if isinstance(t, dict)])
                if not text:
                    continue
                
                if self.tokenizer:
                    tokens = self.tokenizer(text, max_length=self.max_seq_len, truncation=True, padding="max_length", return_tensors="pt")
                    input_ids = tokens["input_ids"].squeeze(0)
                    attention_mask = tokens["attention_mask"].squeeze(0)
                else:
                    input_ids = torch.randint(0, 32000, (self.max_seq_len,))
                    attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)
                
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                    "attention_mask": attention_mask,
                    "images": image,
                    "image_positions": torch.tensor([0]),
                }
            except Exception:
                continue


class MixedStreamDataset(IterableDataset):
    """
    Interleave VL and text-only datasets with configurable ratio.
    
    Implements DeepSeek-VL's 70:30 VL:text mixing strategy.
    """
    
    def __init__(
        self,
        vl_datasets: List[str],
        text_datasets: List[str] = None,
        processor: ProgressiveProcessor = None,
        tokenizer = None,
        max_seq_len: int = 2048,
        text_ratio: float = 0.3,
        vl_probabilities: List[float] = None,
    ):
        self.processor = processor or ProgressiveProcessor()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_ratio = text_ratio
        
        # VL streams
        if vl_probabilities is None:
            vl_probabilities = [1.0 / len(vl_datasets)] * len(vl_datasets)
        self.vl_probabilities = vl_probabilities
        
        self.vl_streams = [
            StreamingVLDataset(ds, self.processor, tokenizer, max_seq_len)
            for ds in vl_datasets
        ]
        
        # Text-only streams
        self.text_streams = []
        if text_datasets and text_ratio > 0:
            self.text_streams = [
                TextOnlyDataset(ds, tokenizer, max_seq_len)
                for ds in text_datasets
            ]
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        vl_iters = [iter(s) for s in self.vl_streams]
        text_iters = [iter(s) for s in self.text_streams] if self.text_streams else []
        
        while True:
            # Decide VL vs text
            use_text = self.text_streams and random.random() < self.text_ratio
            
            if use_text:
                idx = random.randint(0, len(text_iters) - 1)
                try:
                    yield next(text_iters[idx])
                except StopIteration:
                    text_iters[idx] = iter(self.text_streams[idx])
                    try:
                        yield next(text_iters[idx])
                    except StopIteration:
                        continue
            else:
                idx = random.choices(range(len(vl_iters)), weights=self.vl_probabilities)[0]
                try:
                    yield next(vl_iters[idx])
                except StopIteration:
                    vl_iters[idx] = iter(self.vl_streams[idx])
                    try:
                        yield next(vl_iters[idx])
                    except StopIteration:
                        continue


# Legacy alias
MultiStreamDataset = MixedStreamDataset


# =============================================================================
# SFT Dataset
# =============================================================================

class SFTDataset(IterableDataset):
    """Dataset for Supervised Fine-Tuning on instruction data."""
    
    def __init__(
        self,
        dataset_name: str,
        processor: ProgressiveProcessor,
        tokenizer = None,
        max_seq_len: int = 2048,
    ):
        self.dataset_name = dataset_name
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        from datasets import load_dataset
        self.dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    def format_conversation(self, conversations):
        """Format to ChatML style."""
        text = ""
        for turn in conversations:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if role in ["human", "user"]:
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role in ["gpt", "assistant"]:
                text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        return text.strip()
    
    def __iter__(self):
        for sample in self.dataset:
            try:
                convs = sample.get("conversations", sample.get("messages", []))
                text = self.format_conversation(convs)
                
                if not text:
                    continue
                
                image = None
                if "image" in sample and sample["image"]:
                    image = self.processor(sample["image"])
                    if image is None:
                        continue
                
                if self.tokenizer:
                    tokens = self.tokenizer(text, max_length=self.max_seq_len, truncation=True, padding="max_length", return_tensors="pt")
                    input_ids = tokens["input_ids"].squeeze(0)
                    attention_mask = tokens["attention_mask"].squeeze(0)
                else:
                    input_ids = torch.randint(0, 32000, (self.max_seq_len,))
                    attention_mask = torch.ones(self.max_seq_len, dtype=torch.long)
                
                result = {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                    "attention_mask": attention_mask,
                    "images": image,
                    "image_positions": torch.tensor([0]) if image is not None else None,
                }
                
                yield result
            except Exception:
                continue


# =============================================================================
# DPO Dataset
# =============================================================================

class DPODataset(IterableDataset):
    """
    Dataset for Direct Preference Optimization.
    
    Supports RLHF-V and VLFeedback style formats.
    """
    
    def __init__(
        self,
        dataset_name: str,
        processor: ProgressiveProcessor,
        tokenizer = None,
        max_seq_len: int = 2048,
    ):
        self.dataset_name = dataset_name
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        from datasets import load_dataset
        self.dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    def tokenize(self, text: str) -> torch.Tensor:
        if self.tokenizer:
            tokens = self.tokenizer(text, max_length=self.max_seq_len, truncation=True, padding="max_length", return_tensors="pt")
            return tokens["input_ids"].squeeze(0)
        return torch.randint(0, 32000, (self.max_seq_len,))
    
    def __iter__(self):
        for sample in self.dataset:
            try:
                # Get chosen and rejected responses
                prompt = sample.get("prompt", sample.get("question", ""))
                chosen = sample.get("chosen", sample.get("chosen_response", ""))
                rejected = sample.get("rejected", sample.get("rejected_response", ""))
                
                if not prompt or not chosen or not rejected:
                    continue
                
                # Process image if present
                image = None
                if "image" in sample and sample["image"]:
                    image = self.processor(sample["image"])
                
                # Tokenize
                chosen_ids = self.tokenize(f"{prompt}\n{chosen}")
                rejected_ids = self.tokenize(f"{prompt}\n{rejected}")
                
                yield {
                    "input_ids": chosen_ids,  # For compatibility
                    "labels": chosen_ids.clone(),
                    "attention_mask": torch.ones(self.max_seq_len, dtype=torch.long),
                    "chosen_input_ids": chosen_ids,
                    "chosen_labels": chosen_ids.clone(),
                    "rejected_input_ids": rejected_ids,
                    "rejected_labels": rejected_ids.clone(),
                    "images": image,
                    "image_positions": torch.tensor([0]) if image is not None else None,
                }
            except Exception:
                continue


# =============================================================================
# Factory Functions
# =============================================================================

def create_dataset(
    stage: str,
    processor: ProgressiveProcessor = None,
    tokenizer = None,
    max_seq_len: int = 2048,
) -> IterableDataset:
    """
    Create dataset for a training stage.
    
    Args:
        stage: "stage1_alignment", "stage2_pretrain", "stage3_sft", "stage4_dpo"
        processor: Image processor
        tokenizer: HuggingFace tokenizer
        max_seq_len: Maximum sequence length
    
    Returns:
        IterableDataset configured for the stage
    """
    if processor is None:
        processor = ProgressiveProcessor()
    
    config = STAGE_DATASETS.get(stage)
    if config is None:
        raise ValueError(f"Unknown stage: {stage}. Use: {list(STAGE_DATASETS.keys())}")
    
    if stage == "stage4_dpo":
        return DPODataset(
            config["vl_datasets"][0],
            processor,
            tokenizer,
            max_seq_len,
        )
    
    if stage == "stage3_sft":
        return SFTDataset(
            config["vl_datasets"][0],
            processor,
            tokenizer,
            max_seq_len,
        )
    
    return MixedStreamDataset(
        vl_datasets=config["vl_datasets"],
        text_datasets=config.get("text_datasets"),
        processor=processor,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        text_ratio=config.get("text_ratio", 0.0),
    )


# Legacy function
def create_streaming_dataset(
    stage: str = "full",
    processor = None,
    tokenizer = None,
    max_seq_len: int = 2048,
    custom_datasets: List[str] = None,
    probabilities: List[float] = None,
) -> IterableDataset:
    """Legacy function - use create_dataset instead."""
    if processor is None:
        processor = ProgressiveProcessor()
    
    if custom_datasets:
        return MixedStreamDataset(
            vl_datasets=custom_datasets,
            processor=processor,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            vl_probabilities=probabilities,
        )
    
    # Map old stage names
    stage_map = {
        "stage1_alignment": "stage1_alignment",
        "stage2_pretrain": "stage2_pretrain", 
        "stage3_finetune": "stage3_sft",
        "full": "stage2_pretrain",
    }
    return create_dataset(stage_map.get(stage, "stage2_pretrain"), processor, tokenizer, max_seq_len)


def create_sft_dataset(
    datasets: List[str],
    processor = None,
    tokenizer = None,
    max_seq_len: int = 2048,
) -> IterableDataset:
    """Create SFT dataset from instruction-following data."""
    if processor is None:
        processor = ProgressiveProcessor()
    
    return SFTDataset(datasets[0], processor, tokenizer, max_seq_len)
