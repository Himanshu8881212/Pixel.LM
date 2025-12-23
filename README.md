# 🌌 PixelLM: Next-Generation Vision-Language Models

[![Code Quality](https://img.shields.io/badge/Implementation-Research--Grade-brightgreen)](https://github.com/Himanshu8881212/Pixel.LM)
[![Architecture](https://img.shields.io/badge/Architecture-DeepSeek--V3-blue)](https://arxiv.org/abs/2412.19437)
[![Optimization](https://img.shields.io/badge/Optimized-FP8%20%7C%20FA3-orange)](https://github.com/Himanshu8881212/Pixel.LM)

PixelLM is a suite of next-generation Vision-Language Models (VLMs) built using the latest advancements in Large Language Model (LLM) and multimodal research. It combines the efficient architecture of **DeepSeek-V3** (MLA, MoE) with the advanced visual understanding capabilities of **DeepSeek-VL2** and **Qwen2-VL**.

---

## 🚀 Model Variants

| Variant | Hidden | Layers | Experts | Total (LLM) | Active (LLM) | Target Device |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Pixel** | 1024 | 8 | 288/8 | **6.07B** | **339M** | Mobile / Edge |
| **MegaPixel** | 2048 | 16 | 288/8 | **27.05B** | **1.5B** | 1x A100/H100 |
| **GigaPixel** | 4096 | 32 | 288/8 | **168B** | **9.42B** | 8x H100 Cluster |

---

## 💎 State-of-the-Art Architecture

PixelLM integrates industry-standard features from top-tier research labs:

### 🧠 Language Backbone (DeepSeek-V3 Style)

- **MLA (Multi-head Latent Attention)**: Dramatic reduction in KV cache size while maintaining SOTA performance.
- **MoE (Mixture of Experts)**: Aux-loss-free routing with DeepSeek-style shared experts and **Expert Parallelism** support.
- **YaRN Context Extension**: Native support for **128K** context windows using interpolated rotary frequencies.

### 👁️ Vision System (VL2 & Qwen2-VL Style)

- **Dynamic Tiling**: Support for any resolution and aspect ratio via adaptive patching.
- **M-RoPE (Multimodal RoPE)**: Decoupled position embeddings for text (1D), images (2D), and videos (3D).
- **Progressive Resolution**: Training schedule increases resolution (224 → 384 → 512) to match learning complexity.
- **ViT Incremental Learning**: InternVL2-style Stage 1.5 unfreezing to boost OCR and document understanding.

---

## 🛠️ 4-Stage Training Pipeline

| Stage | Focus | Purpose |
| :---: | :--- | :--- |
| **1** | **Alignment** | Mapping visual tokens to LLM embedding space via projector training. |
| **1.5** | **Vision Boost** | Unfreezing the ViT for specialized OCR and document understanding. |
| **2** | **Pretraining** | Large-scale training with 70:30 VL:Text mixing (DeepSeek-VL2 strategy). |
| **3** | **SFT** | Instruction fine-tuning on high-quality conversational and reasoning data. |
| **4** | **Preference** | Alignment via **GRPO** (Grouped Policy Optimization) using the veRL framework. |

---

## ⚡ Production Infrastructure

Designed for massive scale and hardware efficiency:

- **Precision**: Native **FP8** support via NVIDIA TransformerEngine for Hopper GPUs.
- **Parallelism**: Full **3D + EP** Parallelism (Tensor, Pipeline, Data, and Expert Parallelism).
- **Efficiency**: Data Packing eliminates padding waste (30-50% speedup).
- **Stability**: [EMA](file:///src/optimizations.py) (Exponential Moving Average) weights for smoother convergence and superior inference.
- **Attention**: Integrated **Flash Attention 3** for ultra-fast Hopper-optimized computations.

---

## 📦 Getting Started

### Installation

```bash
pip install -r requirements.txt
# For scaled training
pip install deepspeed transformer-engine flash-attn
# For GRPO
pip install verl vllm
```

### Quick Training Start

```bash
# Optimized training with packing and EMA
python train_optimized.py --variant pixel --batch-size 15 --gradient-accumulation 4

# Scaled training via DeepSpeed
deepspeed train_optimized.py --variant megapixel --deepspeed configs/ds_zero3.json

# Scaled training via NeMo
python train_nemo.py --stage stage2 --variant megapixel --num-gpus 8
```

---

## 🧪 Testing

The repository includes a comprehensive test suite for both architecture correctness and production features:

```bash
pytest tests/ -v
pytest tests/test_production.py -v
```

---

## 📝 Citation

If you use PixelLM in your research, please cite:

```bibtex
@misc{ninawe2025pixellm,
  title={PixelLM: Next-Generation Vision-Language Models with Latent Attention},
  author={Himanshu Ninawe},
  year={2025},
  publisher={GitHub},
  journal={GitHub Repository},
  howpublished={\url{https://github.com/Himanshu8881212/Pixel.LM}}
}
```
