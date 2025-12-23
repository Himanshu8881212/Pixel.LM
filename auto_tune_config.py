#!/usr/bin/env python3
"""Tune Pixel to 7B with 32:1 expert, ~16:1 param sparsity."""

def format_num(n):
    if n >= 1e9: return f"{n/1e9:.2f}B"
    elif n >= 1e6: return f"{n/1e6:.0f}M"
    return f"{n/1e3:.0f}K"

def calc_mla(h, heads):
    q_lora, kv_lora = h // 2, h // 8
    return (h * q_lora + q_lora + q_lora * heads * 128 + 
            h * (kv_lora + 64) + kv_lora + kv_lora * heads * 192 + heads * 128 * h)

def calc_moe(h, experts, expert_h, shared_h):
    return h * experts + experts * 3 * h * expert_h + 3 * h * shared_h + h

def calc_lm(h, L, heads, experts, routed, expert_h, shared_h, vocab=102400):
    embed = vocab * h
    mla = calc_mla(h, heads)
    moe = calc_moe(h, experts, expert_h, shared_h)
    per_layer = 2 * h + mla + moe
    total = embed + L * per_layer + h
    active_moe = h * routed + routed * 3 * h * expert_h + 3 * h * shared_h + h
    active = embed + L * (2 * h + mla + active_moe) + h
    return total, active

# Pixel: 7B with 1024×8, 256/8 experts
h, L, experts, routed = 1024, 8, 256, 8
heads = h // 128

print("=" * 50)
print("  PIXEL 7B TUNING")
print("=" * 50)

for exp_h in range(64, 4096, 64):
    shared_h = 2 * h
    total, active = calc_lm(h, L, heads, experts, routed, exp_h, shared_h)
    if abs(total - 7e9) < 0.1e9:
        p_sparse = total / active
        print(f"\nFound: expert_hidden_size={exp_h}")
        print(f"Total: {format_num(total)}, Active: {format_num(active)}")
        print(f"Param sparsity: {p_sparse:.1f}:1")
        print(f'''
    "pixel": ModelConfig(
        variant="pixel",
        vision=VisionConfig(image_size=384, patch_size=16, hidden_size=1024, num_layers=24, num_heads=16, mlp_ratio=4),
        language=LanguageConfig(
            hidden_size=1024, num_layers=8, num_heads=8, head_dim=128,
            intermediate_size=4096, vocab_size=102400, max_position_embeddings=8192,
            mla=MLAConfig(q_lora_rank=512, kv_lora_rank=128, qk_rope_head_dim=64, qk_nope_head_dim=64, v_head_dim=128),
            moe=MoEConfig(enabled=True, num_experts=256, num_experts_per_token=8,
                expert_hidden_size={exp_h}, shared_expert_hidden_size=2048,
                use_shared_expert=True, layer_freq=1),
        ),
    ),''')
        break
