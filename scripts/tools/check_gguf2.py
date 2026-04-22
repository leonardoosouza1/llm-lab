#!/usr/bin/env python3
"""Check multiple GGUF tensor offsets and values against safetensors."""

import struct
import numpy as np
from pathlib import Path

GGUF_PATH = Path("models/qwen2.5-1.5b-q4km.gguf")
ST_PATH = Path("models/qwen2.5-1.5b/model.safetensors")

def read_st(path, name):
    with open(path, "rb") as f:
        hl = struct.unpack("<Q", f.read(8))[0]
        import json
        h = json.loads(f.read(hl))
        m = h[name]
        off = m["data_offsets"]
        f.seek(8 + hl + off[0])
        raw = f.read(off[1] - off[0])
        if m["dtype"] == "BF16":
            u16s = np.frombuffer(raw, dtype=np.uint16)
            f32s = np.zeros(len(u16s), dtype=np.float32)
            f32s.view(np.uint32)[:] = u16s.astype(np.uint32) << 16
            return f32s.reshape(m["shape"])
        elif m["dtype"] == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(m["shape"])
        raise ValueError(m["dtype"])

def parse_gguf(path):
    with open(path, "rb") as f:
        raw = f.read()
    pos = 24
    n_tensors = struct.unpack_from("<Q", raw, 8)[0]
    n_kv = struct.unpack_from("<Q", raw, 16)[0]

    def rs(p):
        l = struct.unpack_from("<Q", raw, p)[0]
        return raw[p+8:p+8+l].decode(), p+8+l

    def rv(p):
        vt = struct.unpack_from("<I", raw, p)[0]; p += 4
        if vt == 4: return struct.unpack_from("<I", raw, p)[0], p+4
        elif vt == 5: return struct.unpack_from("<i", raw, p)[0], p+4
        elif vt == 6: return struct.unpack_from("<f", raw, p)[0], p+4
        elif vt == 7: return raw[p], p+1
        elif vt == 8: return rs(p)
        elif vt == 9:
            et = struct.unpack_from("<I", raw, p)[0]; p += 4
            n = struct.unpack_from("<Q", raw, p)[0]; p += 8
            for _ in range(n):
                _, p = rae(p, et)
            return None, p
        elif vt == 0: return raw[p], p+1
        elif vt == 10: return struct.unpack_from("<Q", raw, p)[0], p+8
        elif vt in (1,2,3,11,12):
            sz = {1:1,2:2,3:2,11:8,12:8}[vt]
            return None, p+sz
        raise ValueError(f"vt={vt}")

    def rae(p, et):
        if et == 4: return struct.unpack_from("<I", raw, p)[0], p+4
        elif et == 5: return struct.unpack_from("<i", raw, p)[0], p+4
        elif et == 6: return struct.unpack_from("<f", raw, p)[0], p+4
        elif et == 8: return rs(p)
        elif et == 0: return raw[p], p+1
        elif et == 10: return struct.unpack_from("<Q", raw, p)[0], p+8
        raise ValueError(f"et={et}")

    meta = {}
    for _ in range(n_kv):
        k, pos = rs(pos)
        v, pos = rv(pos)
        if v is not None: meta[k] = v

    tinfo = {}
    for _ in range(n_tensors):
        name, pos = rs(pos)
        nd = struct.unpack_from("<I", raw, pos)[0]; pos += 4
        shape = [struct.unpack_from("<Q", raw, pos + i*8)[0] for i in range(nd)]
        pos += nd*8
        dt = struct.unpack_from("<I", raw, pos)[0]; pos += 4
        off = struct.unpack_from("<Q", raw, pos)[0]; pos += 8
        tinfo[name] = {"shape": shape, "dtype": dt, "offset": off}

    align = meta.get("general.alignment", 32)
    tdo = ((pos + align - 1) // align) * align
    return raw, tinfo, tdo

def dequant_q6k(raw_bytes, n_elem):
    """Pure Python Q6_K dequant."""
    QK_K = 256
    BS = 210
    n_blocks = n_elem // QK_K
    out = np.zeros(n_elem, dtype=np.float32)

    for i in range(n_blocks):
        off = i * BS
        ql = raw_bytes[off:off+128]
        qh = raw_bytes[off+128:off+192]
        sc = raw_bytes[off+192:off+208]
        d = float(np.frombuffer(raw_bytes[off+208:off+210], dtype=np.float16)[0])

        ql_off = 0; qh_off = 0; sc_off = 0; y_off = i * QK_K
        for _ in range(2):
            for l in range(32):
                q1 = int((ql[ql_off+l] & 0xF) | (((qh[qh_off+l] >> 0) & 3) << 4))
                q2 = int((ql[ql_off+l+32] & 0xF) | (((qh[qh_off+l] >> 2) & 3) << 4))
                q3 = int(((ql[ql_off+l] >> 4) & 0xF) | (((qh[qh_off+l] >> 4) & 3) << 4))
                q4 = int(((ql[ql_off+l+32] >> 4) & 0xF) | (((qh[qh_off+l] >> 6) & 3) << 4))

                # Convert to signed (6-bit value - 32)
                q1 = q1 - 32
                q2 = q2 - 32
                q3 = q3 - 32
                q4 = q4 - 32

                is_idx = l // 16
                s0 = np.array([sc[sc_off + is_idx]], dtype=np.uint8).view(np.int8)[0].item()
                s2 = np.array([sc[sc_off + is_idx + 2]], dtype=np.uint8).view(np.int8)[0].item()
                s4 = np.array([sc[sc_off + is_idx + 4]], dtype=np.uint8).view(np.int8)[0].item()
                s6 = np.array([sc[sc_off + is_idx + 6]], dtype=np.uint8).view(np.int8)[0].item()

                out[y_off + l]      = d * s0 * q1
                out[y_off + l + 32] = d * s2 * q2
                out[y_off + l + 64] = d * s4 * q3
                out[y_off + l + 96] = d * s6 * q4

            ql_off += 64; qh_off += 32; sc_off += 8; y_off += 128

    return out

def dequant_q4k(raw_bytes, n_elem):
    QK_K = 256; BS = 144
    n_blocks = n_elem // QK_K
    out = np.zeros(n_elem, dtype=np.float32)

    def gsm(j, sc):
        if j < 4:
            return (sc[j] & 63, sc[j+4] & 63)
        else:
            d = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4)
            m = (sc[j+4] >> 4) | ((sc[j] >> 6) << 4)
            return (d, m)

    for i in range(n_blocks):
        off = i * BS
        d = float(np.frombuffer(raw_bytes[off:off+2], dtype=np.float16)[0])
        dm = float(np.frombuffer(raw_bytes[off+2:off+4], dtype=np.float16)[0])
        sc = raw_bytes[off+4:off+16]
        qs = raw_bytes[off+16:off+144]
        is_idx = 0; qi = 0; oi = i * QK_K
        for j in range(4):
            s1, m1 = gsm(is_idx, sc)
            s2, m2 = gsm(is_idx+1, sc)
            d1 = d * s1; m1f = dm * m1
            d2 = d * s2; m2f = dm * m2
            for l in range(32):
                out[oi] = d1 * (qs[qi+l] & 0xF) - m1f; oi += 1
            for l in range(32):
                out[oi] = d2 * (qs[qi+l] >> 4) - m2f; oi += 1
            qi += 32; is_idx += 2
    return out

def main():
    raw, tinfo, tdo = parse_gguf(GGUF_PATH)

    # Check multiple tensors
    checks = [
        ("blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight", "Q4_K"),
        ("blk.0.attn_k.weight", "model.layers.0.self_attn.k_proj.weight", "Q4_K"),
        ("blk.0.attn_v.weight", "model.layers.0.self_attn.v_proj.weight", "Q6_K"),
        ("blk.0.attn_output.weight", "model.layers.0.self_attn.o_proj.weight", "Q4_K"),
        ("blk.0.ffn_gate.weight", "model.layers.0.mlp.gate_proj.weight", "Q4_K"),
        ("blk.0.ffn_up.weight", "model.layers.0.mlp.up_proj.weight", "Q4_K"),
        ("blk.0.ffn_down.weight", "model.layers.0.mlp.down_proj.weight", "Q6_K"),
        ("blk.1.attn_q.weight", "model.layers.1.self_attn.q_proj.weight", "Q4_K"),
        ("blk.27.attn_q.weight", "model.layers.27.self_attn.q_proj.weight", "Q4_K"),
    ]

    for gname, stname, qtype in checks:
        ti = tinfo[gname]
        ne = 1
        for s in ti["shape"]: ne *= s

        # Get raw bytes
        start = tdo + ti["offset"]
        if qtype == "Q4_K":
            nb = ne // 256 * 144
            gguf_f32 = dequant_q4k(raw[start:start+nb], ne)
        else:
            nb = ne // 256 * 210
            gguf_f32 = dequant_q6k(raw[start:start+nb], ne)

        # Reshape: GGUF [ne0, ne1] → [ne1, ne0]
        shape = list(reversed(ti["shape"]))
        gguf_f32 = gguf_f32.reshape(shape)

        # Read safetensors
        st_f32 = read_st(ST_PATH, stname)

        # Compare
        diff = np.abs(gguf_f32 - st_f32)
        corr = np.corrcoef(gguf_f32.flatten()[:5000], st_f32.flatten()[:5000])[0,1]

        print(f"{gname:40s} shape={shape} {qtype} | corr={corr:.4f} max_diff={diff.max():.4f} mean_diff={diff.mean():.4f}")

        # Spot check first 5 values
        if "blk.0" in gname:
            g5 = gguf_f32.flatten()[:5]
            s5 = st_f32.flatten()[:5]
            print(f"  GGUF: {g5}")
            print(f"  ST:   {s5}")
            print(f"  diff: {np.abs(g5-s5)}")

if __name__ == "__main__":
    main()
