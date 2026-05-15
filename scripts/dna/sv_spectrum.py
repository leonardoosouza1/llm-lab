#!/usr/bin/env python3
"""SV spectrum analyzer — compares singular value decay for Qwen2.5 vs Qwen3 V_proj.

Hypothesis: Qwen3 V_proj has flatter spectrum → SVD truncation at same rank %
loses more energy → bigger downstream drift.

Usage:
  python sv_spectrum.py
"""
import json
import os
import sys
import numpy as np
from safetensors import safe_open

LAB = "/home/leonardo/projects/LLM/llm-lab"
MODELS = [
    ("qwen2.5-1.5b", "model.safetensors", None),
    ("qwen3-4b",     None,                 "model.safetensors.index.json"),
]
TENSORS = [
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
]

def load_tensor(model_dir, single_file, index_file, name):
    """Load tensor from either single safetensors or sharded model."""
    import torch
    if single_file:
        with safe_open(os.path.join(LAB, "models", model_dir, single_file), framework='pt') as st:
            return st.get_tensor(name).float().numpy()
    else:
        with open(os.path.join(LAB, "models", model_dir, index_file)) as f:
            idx = json.load(f)
        shard = idx['weight_map'][name]
        with safe_open(os.path.join(LAB, "models", model_dir, shard), framework='pt') as st:
            return st.get_tensor(name).float().numpy()

def spectrum(w, top_k=None):
    """Returns singular values, total energy, energy at each rank cutoff."""
    # Convert to float32 if needed
    w_f32 = w.astype(np.float32)
    s = np.linalg.svd(w_f32, compute_uv=False)
    total_energy = (s ** 2).sum()
    return s, total_energy

TARGETS_PCT = [50, 80, 90, 95, 99]
print(f"{'model':<13} {'tensor':<10} {'shape':<14} {'#SV':<6} {'r256%':<7} {' | '.join(f'r{p}%' for p in TARGETS_PCT)}")
print("─" * 100)
for model_dir, single, idx in MODELS:
    for tname in TENSORS:
        try:
            w = load_tensor(model_dir, single, idx, tname)
            s, total = spectrum(w)
            n_sv = len(s)
            energies_cum = np.cumsum(s ** 2)
            r256_energy = energies_cum[min(255, n_sv-1)]
            pct = 100.0 * r256_energy / total
            short_name = tname.replace("model.layers.0.self_attn.", "").replace("model.layers.0.mlp.", "").replace(".weight", "")
            # Find rank needed for each target % energy
            ranks = []
            for target_pct in TARGETS_PCT:
                target_energy = total * target_pct / 100
                # find smallest k such that energies_cum[k-1] >= target_energy
                k = int(np.searchsorted(energies_cum, target_energy)) + 1
                ranks.append(min(k, n_sv))
            print(f"{model_dir:<13} {short_name:<10} {str(w.shape):<14} {n_sv:<6} {pct:<7.1f} {' | '.join(f'{r:>4}' for r in ranks)}")
        except Exception as e:
            print(f"{model_dir} {tname}: ERROR {e}")
