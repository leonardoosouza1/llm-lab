#!/usr/bin/env python3
"""gguf_to_dna5.py — convert a GGUF Q4_K_M file to DNA-V5 (.dna).

Reads ggml-quantized blocks (Q4_K, Q6_K, Q8_0) and packs them into a .dna
container that iara-engine can load via DnaTensor::Block. No re-quantization;
preserves bit-for-bit the llama.cpp output.

DNA-V5 file format (extends DNA-V4):
  header:
    magic       4 bytes b"DNA4"
    version     u32 LE  (5 for DNA-V5)
    n_tensors   u32 LE
  per tensor:
    chunk_magic 4 bytes  (TBLK | TRAW | TSVD)
    name_len    u16 LE
    name        utf-8 bytes
    payload_len u64 LE
    payload     (TBLK = ndim u32 + dims u32× + dtype u32 + raw_block_bytes)
                (TRAW = ndim u32 + dims u32× + dtype u32 + raw_bytes)

dtype codes (match iara-engine/storage/gguf.rs GgmlType):
  0 = BF16  3 = F32  4 = Q4_0  8 = Q8_0  12 = Q4_K  14 = Q6_K

Usage:
  python3 gguf_to_dna5.py --input model-q4km.gguf --output model.dna
"""
import argparse, re, struct, sys, os
import gguf
import numpy as np

# GGUF (ggml) tensor name → HuggingFace name. iara-engine's DNA loader uses HF names.
def gguf_to_hf(name: str) -> str:
    m = re.match(r"^blk\.(\d+)\.(.+)$", name)
    if m:
        idx, suffix = m.group(1), m.group(2)
        suffix_map = {
            "attn_q.weight": "self_attn.q_proj.weight",
            "attn_k.weight": "self_attn.k_proj.weight",
            "attn_v.weight": "self_attn.v_proj.weight",
            "attn_output.weight": "self_attn.o_proj.weight",
            "attn_q_norm.weight": "self_attn.q_norm.weight",
            "attn_k_norm.weight": "self_attn.k_norm.weight",
            "attn_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "ffn_gate.weight": "mlp.gate_proj.weight",
            "ffn_up.weight": "mlp.up_proj.weight",
            "ffn_down.weight": "mlp.down_proj.weight",
        }
        if suffix in suffix_map:
            return f"model.layers.{idx}.{suffix_map[suffix]}"
        return f"model.layers.{idx}.{suffix}"
    flat = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    return flat.get(name, name)

# DNA-V5 dtype codes (must match iara-engine compute-core/dna_format.rs).
DTYPE_BF16 = 0
DTYPE_F16  = 1
DTYPE_F32  = 3
DTYPE_Q4_0 = 4
DTYPE_Q8_0 = 8
DTYPE_Q4_K = 12
DTYPE_Q6_K = 14

# Map gguf GGMLQuantizationType → (dna_dtype, block_size_elements, block_bytes).
GGML_TO_DNA = {
    gguf.GGMLQuantizationType.F32:  (DTYPE_F32,  1,   4),
    gguf.GGMLQuantizationType.F16:  (DTYPE_BF16, 1,   2),  # stored as 2-byte raw; engine widens
    gguf.GGMLQuantizationType.BF16: (DTYPE_BF16, 1,   2),
    gguf.GGMLQuantizationType.Q4_0: (DTYPE_Q4_0, 32,  18),
    gguf.GGMLQuantizationType.Q8_0: (DTYPE_Q8_0, 32,  34),
    gguf.GGMLQuantizationType.Q4_K: (DTYPE_Q4_K, 256, 144),
    gguf.GGMLQuantizationType.Q6_K: (DTYPE_Q6_K, 256, 210),
}


def write_chunk_header(f, magic, name, payload_len):
    f.write(magic)
    nb = name.encode("utf-8")
    f.write(struct.pack("<H", len(nb)))
    f.write(nb)
    f.write(struct.pack("<Q", payload_len))


def write_traw(f, name, shape, dtype, data_bytes):
    payload = struct.pack("<I", len(shape))
    for d in shape:
        payload += struct.pack("<I", d)
    payload += struct.pack("<I", dtype)
    payload += data_bytes
    write_chunk_header(f, b"TRAW", name, len(payload))
    f.write(payload)


def write_tblk(f, name, shape, dtype, blocks):
    payload = struct.pack("<I", len(shape))
    for d in shape:
        payload += struct.pack("<I", d)
    payload += struct.pack("<I", dtype)
    payload += blocks
    write_chunk_header(f, b"TBLK", name, len(payload))
    f.write(payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="GGUF Q4_K_M file")
    ap.add_argument("--output", required=True, help="DNA-V5 output (.dna)")
    args = ap.parse_args()

    reader = gguf.GGUFReader(args.input)
    tensors = reader.tensors
    print(f"opened {args.input}: {len(tensors)} tensors")

    # Detect tied embeddings (no output.weight tensor → tied to token_embd).
    has_output_weight = any(t.name == "output.weight" for t in tensors)
    has_token_embd = any(t.name == "token_embd.weight" for t in tensors)
    synth_lm_head = (not has_output_weight) and has_token_embd
    if synth_lm_head:
        print("  tied embeddings detected → synthesizing lm_head.weight from token_embd")

    # Sort by HF name for deterministic output (engine looks up by HF name).
    tensors = sorted(tensors, key=lambda t: gguf_to_hf(t.name))

    counts = {"q4k": 0, "q6k": 0, "q8_0": 0, "f16": 0, "f32": 0, "skip": 0}

    with open(args.output, "wb") as f:
        f.write(b"DNA4")
        f.write(struct.pack("<I", 5))        # version 5
        count_pos = f.tell()
        f.write(struct.pack("<I", 0))        # placeholder n_tensors

        n_written = 0
        embed_raw_for_lm_head = None
        embed_shape = None
        embed_dna_dtype = None
        for t in tensors:
            ggml_dtype = t.tensor_type
            entry = GGML_TO_DNA.get(ggml_dtype)
            if entry is None:
                print(f"  skip {t.name} (dtype={ggml_dtype})")
                counts["skip"] += 1
                continue
            dna_dtype, block_size, block_bytes = entry

            # Translate ggml name → HF name expected by iara-engine DNA loader.
            hf_name = gguf_to_hf(t.name)

            # GGUF shape is [ne0, ne1, ne2, ne3]; ne0 = innermost. PyTorch nn.Linear
            # weights are [out, in] which GGUF stores as [in, out] (ne0=in, ne1=out).
            # We need to write [out, in] so engine's LayerWeight expects n=out, k=in.
            shape_raw = [int(x) for x in t.shape if x > 0]
            if len(shape_raw) == 2:
                shape = [shape_raw[1], shape_raw[0]]
            else:
                shape = shape_raw

            numel = 1
            for d in shape:
                numel *= d

            # Reconstruct raw bytes from the in-memory tensor.
            raw = bytes(t.data.tobytes())

            # Stash embed for synthetic lm_head (same Q6_K blocks). lm_head doesn't
            # require dequant — engine maps DnaTensor::Block Q6_K → LayerWeight::Q6K
            # → gemv_q6k path direto (mesmo que GGUF tied).
            if hf_name == "model.embed_tokens.weight":
                embed_raw_for_lm_head = raw
                embed_shape = list(shape)
                embed_dna_dtype = dna_dtype

            # Engine embed_lookup hardcoded BF16 u16. Dequant Q4_K/Q6_K embed →
            # BF16 → write TRAW BF16 (engine take_full retorna AnyTensor BF16).
            # lm_head pode ficar Q6_K porque LayerWeight::Q6K kernel suporta.
            if hf_name == "model.embed_tokens.weight" and \
               ggml_dtype in (gguf.GGMLQuantizationType.Q4_K,
                              gguf.GGMLQuantizationType.Q6_K,
                              gguf.GGMLQuantizationType.Q8_0):
                f32 = gguf.quants.dequantize(t.data, ggml_dtype).astype(np.float32)
                f32_u32 = f32.view(np.uint32)
                bf16_u16 = (f32_u32 >> 16).astype(np.uint16)
                write_traw(f, hf_name, shape, DTYPE_BF16, bf16_u16.tobytes())
                counts["f16"] += 1
                n_written += 1
                continue

            if ggml_dtype in (gguf.GGMLQuantizationType.Q4_K,
                              gguf.GGMLQuantizationType.Q6_K,
                              gguf.GGMLQuantizationType.Q8_0):
                # Block-quantized: validate size and write TBLK.
                n_blocks = (numel + block_size - 1) // block_size
                expected_bytes = n_blocks * block_bytes
                if len(raw) != expected_bytes:
                    print(f"  WARN {hf_name}: bytes={len(raw)} expected={expected_bytes} "
                          f"(numel={numel}, block_size={block_size}, block_bytes={block_bytes})")
                write_tblk(f, hf_name, shape, dna_dtype, raw)
                if ggml_dtype == gguf.GGMLQuantizationType.Q4_K:
                    counts["q4k"] += 1
                elif ggml_dtype == gguf.GGMLQuantizationType.Q6_K:
                    counts["q6k"] += 1
                else:
                    counts["q8_0"] += 1
            elif ggml_dtype == gguf.GGMLQuantizationType.F32:
                write_traw(f, hf_name, shape, DTYPE_F32, raw)
                counts["f32"] += 1
            elif ggml_dtype in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
                write_traw(f, hf_name, shape, DTYPE_BF16, raw)
                counts["f16"] += 1
            else:
                counts["skip"] += 1
                continue
            n_written += 1

        # Synthesize lm_head.weight as a TBLK copy of embed (tied embeddings).
        # Engine load_dna requires explicit lm_head tensor; same approach used by
        # iara-train/bin/dna_compress for safetensors → DNA-V4.
        if synth_lm_head and embed_raw_for_lm_head is not None:
            print(f"  writing synthetic lm_head.weight ({embed_shape}, dtype={embed_dna_dtype})")
            if embed_dna_dtype in (DTYPE_Q4_K, DTYPE_Q6_K, DTYPE_Q8_0):
                write_tblk(f, "lm_head.weight", embed_shape, embed_dna_dtype, embed_raw_for_lm_head)
            else:
                write_traw(f, "lm_head.weight", embed_shape, embed_dna_dtype, embed_raw_for_lm_head)
            n_written += 1

        # Patch n_tensors header.
        f.seek(count_pos)
        f.write(struct.pack("<I", n_written))

    sz = os.path.getsize(args.output)
    print(f"wrote {args.output} ({sz/1e9:.2f} GB, {n_written} tensors)")
    print(f"  Q4_K={counts['q4k']}  Q6_K={counts['q6k']}  Q8_0={counts['q8_0']}  "
          f"F16/BF16={counts['f16']}  F32={counts['f32']}  skip={counts['skip']}")


if __name__ == "__main__":
    main()
