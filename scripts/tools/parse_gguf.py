#!/usr/bin/env python3
"""Parse a GGUF v3 file and print metadata, tensor info, and summary."""

import struct
import sys
from collections import Counter

GGUF_PATH = "/home/leonardo/.ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4"

# GGML type names and their byte-per-element info (block_size, type_size)
GGML_TYPE_NAME = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q5_K", 15: "Q6_K", 16: "Q8_K", 26: "IQ4_NL",
    30: "BF16",
}

# (block_size, type_size_in_bytes) for computing tensor data size
GGML_TYPE_SIZE = {
    0:  (1, 4),      # F32
    1:  (1, 2),      # F16
    2:  (32, 18),    # Q4_0
    3:  (32, 20),    # Q4_1
    6:  (32, 22),    # Q5_0
    7:  (32, 24),    # Q5_1
    8:  (32, 34),    # Q8_0
    9:  (32, 36),    # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 176),  # Q5_K
    15: (256, 210),  # Q6_K
    16: (256, 292),  # Q8_K
    26: (32, 18),    # IQ4_NL
    30: (1, 2),      # BF16
}

GGUFValueType = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16",
    4: "uint32", 5: "int32", 6: "float32", 7: "bool",
    8: "string", 9: "array", 10: "uint64", 11: "int64", 12: "float64",
}

def read_u8(f):  return struct.unpack("<B", f.read(1))[0]
def read_i8(f):  return struct.unpack("<b", f.read(1))[0]
def read_u16(f): return struct.unpack("<H", f.read(2))[0]
def read_i16(f): return struct.unpack("<h", f.read(2))[0]
def read_u32(f): return struct.unpack("<I", f.read(4))[0]
def read_i32(f): return struct.unpack("<i", f.read(4))[0]
def read_u64(f): return struct.unpack("<Q", f.read(8))[0]
def read_i64(f): return struct.unpack("<q", f.read(8))[0]
def read_f32(f): return struct.unpack("<f", f.read(4))[0]
def read_f64(f): return struct.unpack("<d", f.read(8))[0]

def read_string(f):
    length = read_u64(f)
    return f.read(length).decode("utf-8", errors="replace")

def read_value(f, vtype):
    if vtype == 0:   return read_u8(f)
    elif vtype == 1:  return read_i8(f)
    elif vtype == 2:  return read_u16(f)
    elif vtype == 3:  return read_i16(f)
    elif vtype == 4:  return read_u32(f)
    elif vtype == 5:  return read_i32(f)
    elif vtype == 6:  return read_f32(f)
    elif vtype == 7:  return bool(read_u8(f))
    elif vtype == 8:  return read_string(f)
    elif vtype == 9:  # array
        elem_type = read_u32(f)
        n_elems = read_u64(f)
        return [read_value(f, elem_type) for _ in range(n_elems)]
    elif vtype == 10: return read_u64(f)
    elif vtype == 11: return read_i64(f)
    elif vtype == 12: return read_f64(f)
    else:
        raise ValueError(f"Unknown value type: {vtype}")

def tensor_data_size(shape, ggml_type):
    n_elements = 1
    for d in shape:
        n_elements *= d
    if ggml_type in GGML_TYPE_SIZE:
        block_size, type_size = GGML_TYPE_SIZE[ggml_type]
        return (n_elements // block_size) * type_size
    return n_elements * 4  # fallback assume f32

def format_value(v):
    if isinstance(v, list):
        if len(v) <= 20:
            return str(v)
        return f"[{v[0]}, {v[1]}, ... ({len(v)} items total) ..., {v[-2]}, {v[-1]}]"
    if isinstance(v, str) and len(v) > 200:
        return v[:200] + f"... ({len(v)} chars)"
    return repr(v)

def main():
    with open(GGUF_PATH, "rb") as f:
        # --- Header ---
        magic = f.read(4)
        if magic != b"GGUF":
            print(f"ERROR: Not a GGUF file (magic: {magic})")
            sys.exit(1)
        version = read_u32(f)
        n_tensors = read_u64(f)
        n_kv = read_u64(f)
        print(f"{'='*70}")
        print(f"GGUF Header")
        print(f"{'='*70}")
        print(f"  Magic:      {magic}")
        print(f"  Version:    {version}")
        print(f"  N tensors:  {n_tensors}")
        print(f"  N KV pairs: {n_kv}")
        print()

        # --- KV Pairs ---
        print(f"{'='*70}")
        print(f"KV Pairs ({n_kv})")
        print(f"{'='*70}")
        for i in range(n_kv):
            key = read_string(f)
            vtype = read_u32(f)
            value = read_value(f, vtype)
            type_name = GGUFValueType.get(vtype, f"unknown({vtype})")
            print(f"  [{i:2d}] {key} ({type_name}) = {format_value(value)}")
        print()

        # --- Tensor Infos ---
        print(f"{'='*70}")
        print(f"Tensor Infos ({n_tensors})")
        print(f"{'='*70}")
        tensors = []
        for i in range(n_tensors):
            name = read_string(f)
            n_dims = read_u32(f)
            dims = [read_u64(f) for _ in range(n_dims)]
            ggml_type = read_u32(f)
            offset = read_u64(f)
            type_name = GGML_TYPE_NAME.get(ggml_type, f"unknown({ggml_type})")
            size = tensor_data_size(dims, ggml_type)
            tensors.append((name, dims, ggml_type, type_name, offset, size))

        def fmt_tensor(i, t):
            name, dims, _, type_name, offset, size = t
            shape_str = " x ".join(str(d) for d in dims)
            if size >= 1024*1024:
                size_str = f"{size / (1024*1024):.1f} MB"
            elif size >= 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  [{i:3d}] {name:<55s} {type_name:<8s} [{shape_str}]  {size_str}")

        print("  First 10 tensors:")
        for i in range(min(10, len(tensors))):
            fmt_tensor(i, tensors[i])

        if len(tensors) > 15:
            print(f"\n  ... ({len(tensors) - 15} tensors omitted) ...\n")
            print("  Last 5 tensors:")
            for i in range(len(tensors)-5, len(tensors)):
                fmt_tensor(i, tensors[i])
        print()

        # --- Summary ---
        print(f"{'='*70}")
        print(f"Summary")
        print(f"{'='*70}")
        type_counter = Counter()
        total_size = 0
        for name, dims, ggml_type, type_name, offset, size in tensors:
            type_counter[type_name] += 1
            total_size += size

        print(f"  Total tensors: {len(tensors)}")
        print(f"  Total tensor data size: {total_size / (1024**3):.2f} GB")
        print(f"\n  Tensors per type:")
        for tname, count in type_counter.most_common():
            print(f"    {tname:<12s}: {count}")

if __name__ == "__main__":
    main()
