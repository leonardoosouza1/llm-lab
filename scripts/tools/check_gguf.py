#!/usr/bin/env python3
"""Compare GGUF dequantized weights against safetensors BF16 weights."""

import struct
import numpy as np
from pathlib import Path

GGUF_PATH = Path("models/qwen2.5-1.5b-q4km.gguf")
ST_PATH = Path("models/qwen2.5-1.5b/model.safetensors")

def read_safetensors_tensor(path, name):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        import json
        header = json.loads(f.read(header_len))
        meta = header[name]
        dtype = meta["dtype"]
        shape = meta["shape"]
        offsets = meta["data_offsets"]
        f.seek(8 + header_len + offsets[0])
        raw = f.read(offsets[1] - offsets[0])
        if dtype == "BF16":
            u16s = np.frombuffer(raw, dtype=np.uint16)
            f32s = np.zeros(len(u16s), dtype=np.float32)
            f32s.view(np.uint32)[:] = u16s.astype(np.uint32) << 16
            return f32s.reshape(shape)
        elif dtype == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(shape)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

def dequant_q4_k_manual(raw_bytes, n_elem):
    """Pure Python Q4_K dequant matching GGML reference."""
    QK_K = 256
    BLOCK_SIZE = 144
    n_blocks = n_elem // QK_K
    output = np.zeros(n_elem, dtype=np.float32)

    for i in range(n_blocks):
        offset = i * BLOCK_SIZE

        # Read d and dmin as FP16
        d = struct.unpack_from('<e', raw_bytes, offset)[0]
        dmin = struct.unpack_from('<e', raw_bytes, offset + 2)[0]

        scales_off = offset + 4
        qs_off = offset + 16

        is_idx = 0
        q_idx = 0
        out_idx = i * QK_K

        for j in range(4):
            # get_scale_min_k4
            if is_idx < 4:
                sc1 = raw_bytes[scales_off + is_idx] & 63
                m1 = raw_bytes[scales_off + is_idx + 4] & 63
            else:
                sc1 = (raw_bytes[scales_off + is_idx + 4] & 0x0F) | ((raw_bytes[scales_off + is_idx - 4] >> 6) << 4)
                m1 = (raw_bytes[scales_off + is_idx + 4] >> 4) | ((raw_bytes[scales_off + is_idx] >> 6) << 4)

            if is_idx + 1 < 4:
                sc2 = raw_bytes[scales_off + is_idx + 1] & 63
                m2 = raw_bytes[scales_off + is_idx + 1 + 4] & 63
            else:
                sc2 = (raw_bytes[scales_off + is_idx + 1 + 4] & 0x0F) | ((raw_bytes[scales_off + is_idx + 1 - 4] >> 6) << 4)
                m2 = (raw_bytes[scales_off + is_idx + 1 + 4] >> 4) | ((raw_bytes[scales_off + is_idx + 1] >> 6) << 4)

            d1 = d * sc1
            m1f = dmin * m1
            d2 = d * sc2
            m2f = dmin * m2

            for l in range(32):
                q = raw_bytes[qs_off + q_idx + l] & 0x0F
                output[out_idx] = d1 * q - m1f
                out_idx += 1

            for l in range(32):
                q = raw_bytes[qs_off + q_idx + l] >> 4
                output[out_idx] = d2 * q - m2f
                out_idx += 1

            q_idx += 32
            is_idx += 2

    return output

def main():
    print("=" * 60)
    print("GGUF vs Safetensors Tensor Comparison")
    print("=" * 60)

    # Load safetensors
    print("\n── Safetensors ──")
    st_qw = read_safetensors_tensor(ST_PATH, "model.layers.0.self_attn.q_proj.weight")
    print(f"q_proj.weight shape: {st_qw.shape}")
    print(f"  first10: {st_qw.flatten()[:10]}")
    print(f"  mean={st_qw.mean():.6f} std={st_qw.std():.6f}")

    st_norm = read_safetensors_tensor(ST_PATH, "model.layers.0.input_layernorm.weight")
    print(f"input_layernorm shape: {st_norm.shape} first10: {st_norm.flatten()[:10]}")

    # Load GGUF raw
    print("\n── GGUF (manual parse) ──")
    with open(GGUF_PATH, "rb") as f:
        raw = f.read()

    # Parse GGUF header to find tensor data
    magic = struct.unpack_from("<I", raw, 0)[0]
    version = struct.unpack_from("<I", raw, 4)[0]
    n_tensors = struct.unpack_from("<Q", raw, 8)[0]
    n_kv = struct.unpack_from("<Q", raw, 16)[0]
    print(f"GGUF v{version}: {n_tensors} tensors, {n_kv} KV pairs")

    pos = 24

    # Skip KV pairs
    def read_str(pos):
        slen = struct.unpack_from("<Q", raw, pos)[0]
        pos += 8
        s = raw[pos:pos+slen].decode("utf-8")
        pos += slen
        return s, pos

    def read_value(pos):
        vtype = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        if vtype == 4:  # u32
            v = struct.unpack_from("<I", raw, pos)[0]
            return v, pos + 4
        elif vtype == 5:  # i32
            v = struct.unpack_from("<i", raw, pos)[0]
            return v, pos + 4
        elif vtype == 6:  # f32
            v = struct.unpack_from("<f", raw, pos)[0]
            return v, pos + 4
        elif vtype == 7:  # bool
            v = raw[pos]
            return bool(v), pos + 1
        elif vtype == 8:  # string
            return read_str(pos)
        elif vtype == 9:  # array
            elem_type = struct.unpack_from("<I", raw, pos)[0]
            pos += 4
            n = struct.unpack_from("<Q", raw, pos)[0]
            pos += 8
            arr = []
            for _ in range(n):
                v, pos = read_array_elem(pos, elem_type)
                arr.append(v)
            return arr, pos
        elif vtype == 0:  # u8
            return raw[pos], pos + 1
        elif vtype == 1:  # i8
            return struct.unpack_from("<b", raw, pos)[0], pos + 1
        elif vtype == 2:  # u16
            return struct.unpack_from("<H", raw, pos)[0], pos + 2
        elif vtype == 3:  # i16
            return struct.unpack_from("<h", raw, pos)[0], pos + 2
        elif vtype == 10:  # u64
            return struct.unpack_from("<Q", raw, pos)[0], pos + 8
        elif vtype == 11:  # i64
            return struct.unpack_from("<q", raw, pos)[0], pos + 8
        elif vtype == 12:  # f64
            return struct.unpack_from("<d", raw, pos)[0], pos + 8
        else:
            raise ValueError(f"Unknown value type: {vtype} at pos {pos-4}")

    def read_array_elem(pos, elem_type):
        if elem_type == 4:
            return struct.unpack_from("<I", raw, pos)[0], pos + 4
        elif elem_type == 5:
            return struct.unpack_from("<i", raw, pos)[0], pos + 4
        elif elem_type == 6:
            return struct.unpack_from("<f", raw, pos)[0], pos + 4
        elif elem_type == 8:
            return read_str(pos)
        elif elem_type == 0:
            return raw[pos], pos + 1
        elif elem_type == 10:
            return struct.unpack_from("<Q", raw, pos)[0], pos + 8
        else:
            raise ValueError(f"Unknown array elem type: {elem_type}")

    # Read KV pairs
    metadata = {}
    for _ in range(n_kv):
        key, pos = read_str(pos)
        val, pos = read_value(pos)
        metadata[key] = val

    alignment = metadata.get("general.alignment", 32)

    # Read tensor infos
    tensor_infos = {}
    for _ in range(n_tensors):
        name, pos = read_str(pos)
        n_dims = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        shape = []
        for _ in range(n_dims):
            shape.append(struct.unpack_from("<Q", raw, pos)[0])
            pos += 8
        dtype = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        offset = struct.unpack_from("<Q", raw, pos)[0]
        pos += 8
        tensor_infos[name] = {"shape": shape, "dtype": dtype, "offset": offset}

    # Compute data section start
    tensor_data_offset = ((pos + alignment - 1) // alignment) * alignment
    print(f"tensor_data_offset = {tensor_data_offset:#x}")

    # Now read and dequantize blk.0.attn_q.weight (Q4_K = type 12)
    qw_info = tensor_infos["blk.0.attn_q.weight"]
    print(f"\nblk.0.attn_q.weight: shape={qw_info['shape']} dtype={qw_info['dtype']}")
    print(f"  offset={qw_info['offset']:#x} abs_offset={tensor_data_offset + qw_info['offset']:#x}")

    n_elem = 1
    for s in qw_info["shape"]:
        n_elem *= s
    print(f"  n_elem={n_elem}")

    # Q4_K: 144 bytes per 256 elements
    n_blocks = n_elem // 256
    n_bytes = n_blocks * 144
    data_start = tensor_data_offset + qw_info["offset"]
    qw_raw = raw[data_start:data_start + n_bytes]

    gguf_qw = dequant_q4_k_manual(qw_raw, n_elem)
    # GGUF shape [ne0=1536, ne1=1536] → PyTorch [ne1, ne0] = [1536, 1536]
    gguf_qw = gguf_qw.reshape(qw_info["shape"][1], qw_info["shape"][0])
    print(f"  dequant shape: {gguf_qw.shape}")
    print(f"  first10: {gguf_qw.flatten()[:10]}")
    print(f"  mean={gguf_qw.mean():.6f} std={gguf_qw.std():.6f}")

    # Compare
    diff = np.abs(gguf_qw - st_qw)
    print(f"\n  ── COMPARISON Q_PROJ ──")
    print(f"  max_abs_diff = {diff.max():.6f}")
    print(f"  mean_abs_diff = {diff.mean():.6f}")
    print(f"  GGUF row0[:10]  = {gguf_qw[0, :10]}")
    print(f"  ST   row0[:10]  = {st_qw[0, :10]}")
    print(f"  row0 diff[:10]  = {diff[0, :10]}")

    # Check transposed
    trans_diff = np.abs(gguf_qw.T - st_qw)
    print(f"  max_abs_diff(transposed) = {trans_diff.max():.6f}")
    print(f"  mean_abs_diff(transposed) = {trans_diff.mean():.6f}")

    # Correlation check
    corr_normal = np.corrcoef(gguf_qw.flatten()[:1000], st_qw.flatten()[:1000])[0, 1]
    corr_trans = np.corrcoef(gguf_qw.T.flatten()[:1000], st_qw.flatten()[:1000])[0, 1]
    print(f"  corr(normal) = {corr_normal:.6f}")
    print(f"  corr(transposed) = {corr_trans:.6f}")

    # Check norm (F32)
    norm_info = tensor_infos["blk.0.attn_norm.weight"]
    print(f"\n── NORM comparison ──")
    norm_start = tensor_data_offset + norm_info["offset"]
    norm_n = norm_info["shape"][0]
    norm_data = np.frombuffer(raw[norm_start:norm_start + norm_n * 4], dtype=np.float32)
    print(f"  GGUF first10: {norm_data[:10]}")
    print(f"  ST   first10: {st_norm.flatten()[:10]}")
    print(f"  diff first10: {np.abs(norm_data[:10] - st_norm.flatten()[:10])}")

    # Check bias (F32)
    for bias_name, st_name in [
        ("blk.0.attn_q.bias", "model.layers.0.self_attn.q_proj.bias"),
        ("blk.0.attn_k.bias", "model.layers.0.self_attn.k_proj.bias"),
        ("blk.0.attn_v.bias", "model.layers.0.self_attn.v_proj.bias"),
    ]:
        if bias_name in tensor_infos:
            bi = tensor_infos[bias_name]
            bias_start = tensor_data_offset + bi["offset"]
            n = bi["shape"][0]
            bias_data = np.frombuffer(raw[bias_start:bias_start + n * 4], dtype=np.float32)
            print(f"\n  {bias_name}: first5={bias_data[:5]}")
            try:
                st_bias = read_safetensors_tensor(ST_PATH, st_name)
                print(f"  {st_name}: first5={st_bias.flatten()[:5]}")
                print(f"  diff: {np.abs(bias_data[:5] - st_bias.flatten()[:5])}")
            except KeyError:
                print(f"  ({st_name} not in safetensors)")

    # Check if output.weight exists
    if "output.weight" in tensor_infos:
        oi = tensor_infos["output.weight"]
        print(f"\noutput.weight: shape={oi['shape']} dtype={oi['dtype']}")
    else:
        print("\noutput.weight: NOT FOUND → tied with token_embd")

if __name__ == "__main__":
    main()
