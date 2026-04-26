#!/usr/bin/env python3
"""Extract kernarg layouts from .hsaco files (Clang Offload Bundles).
Emits a Rust source file with a compile-time HashMap of kernel name -> KernelLayout.

Usage: extract_layouts.py <out_rs> <hsaco1> [hsaco2 ...]
"""
import os
import re
import struct
import subprocess
import sys
from pathlib import Path


def extract_amdgcn_elf(data: bytes) -> bytes | None:
    if data[:24] != b"__CLANG_OFFLOAD_BUNDLE__":
        return data  # already raw ELF
    num = struct.unpack_from("<Q", data, 24)[0]
    p = 32
    for _ in range(num):
        off, sz, idl = struct.unpack_from("<QQQ", data, p)
        p += 24
        idstr = data[p : p + idl].decode("utf-8", errors="ignore")
        p += idl
        if "amdgcn" in idstr:
            return data[off : off + sz]
    return None


def parse_kernel_blocks(yaml_text: str):
    """Return dict[name] = list of (offset, size, value_kind)."""
    # Split blocks by '.name:'.  YAML format from llvm-readelf is mostly flat per kernel.
    blocks = re.split(r"^\s*-\s+\.args:\s*$", yaml_text, flags=re.M)[1:]
    # Each block is followed by .args content then trailing kernel-level fields up to next kernel start.
    # Easier path: process the whole text linearly.
    out = {}
    cur_name = None
    args_pending = []  # collected args until we see .name
    in_args = False
    arg_offset = None
    arg_size = None
    arg_kind = None
    pending_args_for_unknown_name = []

    # Stream: walk lines; collect args between - .actual_access/.address_space/.offset/.size/.value_kind.
    lines = yaml_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("- .args:") or stripped == "- .args:":
            # Start collecting args for upcoming kernel
            args_pending = []
            args_pending_active = True
            i += 1
            # Walk subsequent argument items until kernarg_segment_size or non-arg key
            current = {}
            while i < len(lines):
                line2 = lines[i]
                m = re.match(r"^\s+-\s+(\.[a-z_]+):\s*(\S+)\s*$", line2)
                if m and m.group(1) == ".actual_access" or (m and m.group(1) in (".actual_access", ".address_space")):
                    if current:
                        args_pending.append(current)
                    current = {}
                    key = m.group(1)
                    current[key] = m.group(2)
                    i += 1
                    continue
                m2 = re.match(r"^\s+(\.[a-z_]+):\s*(\S+)\s*$", line2)
                if m2:
                    key = m2.group(1)
                    val = m2.group(2)
                    if key == ".offset":
                        if "offset" in current:
                            args_pending.append(current)
                            current = {}
                        current["offset"] = int(val)
                    elif key == ".size":
                        current["size"] = int(val)
                    elif key == ".value_kind":
                        current["value_kind"] = val
                    elif key == ".address_space":
                        current.setdefault("address_space", val)
                    elif key == ".actual_access":
                        current.setdefault("actual_access", val)
                    elif key == ".name":
                        if current:
                            args_pending.append(current)
                        out[val] = list(args_pending)
                        args_pending = []
                        current = {}
                    elif key == ".kernarg_segment_size":
                        # ignore here, we'll re-derive
                        pass
                    elif key in (".symbol",):
                        pass
                    elif key in (".group_segment_fixed_size", ".private_segment_fixed_size",
                                ".sgpr_count", ".sgpr_spill_count", ".vgpr_count", ".vgpr_spill_count",
                                ".max_flat_workgroup_size", ".kernarg_segment_align",
                                ".language_version", ".language", ".uniform_work_group_size",
                                ".uses_dynamic_stack", ".wavefront_size", ".workgroup_processor_mode"):
                        pass
                    i += 1
                    continue
                # Unknown line breaks arg collection
                break
        else:
            i += 1

    # Filter to only "by_value" or "global_buffer" (we ignore hidden_* args; they sit at end and zero-fill works).
    return out


def kind_to_rust(kind: str) -> str:
    if kind == "global_buffer":
        return "ArgKind::Pointer"
    if kind == "by_value":
        return "ArgKind::Scalar"
    if kind.startswith("hidden_"):
        return "ArgKind::Hidden"
    return "ArgKind::Hidden"  # treat unknowns as hidden (zeroed)


def main():
    if len(sys.argv) < 3:
        print("usage: extract_layouts.py <out_rs> <hsaco1> [hsaco2 ...]", file=sys.stderr)
        sys.exit(2)

    out_rs = Path(sys.argv[1])
    hsaco_paths = [Path(p) for p in sys.argv[2:]]

    rocm = os.environ.get("ROCM_PATH", "/opt/rocm-7.2.1")
    readelf = f"{rocm}/lib/llvm/bin/llvm-readelf"
    if not Path(readelf).exists():
        readelf = "llvm-readelf"

    layouts = {}
    kernarg_size_total = {}

    for hpath in hsaco_paths:
        data = hpath.read_bytes()
        elf = extract_amdgcn_elf(data)
        if elf is None:
            continue
        elf_path = Path(str(hpath) + ".elf.tmp")
        elf_path.write_bytes(elf)
        try:
            res = subprocess.run([readelf, "--notes", str(elf_path)],
                                 capture_output=True, text=True, check=True)
            yaml_text = res.stdout
        finally:
            elf_path.unlink(missing_ok=True)

        # Parse manually: walk lines.
        cur_name = None
        cur_args = []
        cur_arg = None
        cur_kernarg_size = 0
        for line in yaml_text.splitlines():
            s = line.strip()
            m_offset = re.match(r"\.offset:\s+(\d+)", s)
            m_size = re.match(r"\.size:\s+(\d+)", s)
            m_kind = re.match(r"\.value_kind:\s+(\S+)", s)
            m_name = re.match(r"\.name:\s+(\S+)", s)
            m_seg = re.match(r"\.kernarg_segment_size:\s+(\d+)", s)
            if s.startswith("- .actual_access") or s.startswith("- .address_space") or s.startswith("- .offset"):
                if cur_arg is not None:
                    cur_args.append(cur_arg)
                cur_arg = {}
                if s.startswith("- .offset"):
                    m = re.match(r"-\s+\.offset:\s+(\d+)", s)
                    if m:
                        cur_arg["offset"] = int(m.group(1))
                continue
            if m_offset:
                if cur_arg is None:
                    cur_arg = {}
                cur_arg["offset"] = int(m_offset.group(1))
            elif m_size and cur_arg is not None:
                cur_arg["size"] = int(m_size.group(1))
            elif m_kind and cur_arg is not None:
                cur_arg["value_kind"] = m_kind.group(1)
            elif m_seg:
                cur_kernarg_size = int(m_seg.group(1))
            elif m_name:
                if cur_arg is not None:
                    cur_args.append(cur_arg)
                    cur_arg = None
                name = m_name.group(1)
                # Skip if "actual_access" or other random; keep proper kernel names.
                if cur_args and name not in ("read_only", "global", "by_value", "OpenCL", "C"):
                    layouts[name] = list(cur_args)
                    kernarg_size_total[name] = cur_kernarg_size
                cur_args = []
                cur_kernarg_size = 0

    # Emit Rust file.
    lines = [
        "// Auto-generated by hip-ffi/scripts/extract_layouts.py — DO NOT EDIT.",
        "// Kernarg layouts extracted from .note.AMDGPU msgpack at build time.",
        "",
        "use std::collections::HashMap;",
        "",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq)]",
        "pub enum HiddenKind {",
        "    BlockCountX, BlockCountY, BlockCountZ,",
        "    GroupSizeX, GroupSizeY, GroupSizeZ,",
        "    RemainderX, RemainderY, RemainderZ,",
        "    GridDims,",
        "    DynamicLdsSize,",
        "    Zero,",
        "}",
        "",
        "#[derive(Clone, Copy, Debug, PartialEq, Eq)]",
        "pub enum ArgKind { Pointer, Scalar, Hidden(HiddenKind) }",
        "",
        "#[derive(Clone, Copy, Debug)]",
        "pub struct ArgSlot { pub offset: u32, pub size: u32, pub kind: ArgKind }",
        "",
        "#[derive(Clone, Debug)]",
        "pub struct KernelLayout {",
        "    pub kernarg_size: u32,",
        "    pub args: &'static [ArgSlot],",
        "}",
        "",
    ]

    for name in sorted(layouts.keys()):
        args = layouts[name]
        ksize = kernarg_size_total.get(name, 0)
        # Emit explicit args (Pointer/Scalar) AND hidden args (with their kind tag).
        # Hidden args must be populated at dispatch time with runtime values for
        # HIP-compiled kernels to work via raw AQL.
        rs_slots = []
        for a in args:
            if "offset" not in a or "size" not in a:
                continue
            kind = a.get("value_kind", "")
            if kind == "global_buffer":
                rk = "ArgKind::Pointer"
            elif kind == "by_value":
                rk = "ArgKind::Scalar"
            elif kind.startswith("hidden_"):
                # Map hidden kind to a tag the runtime can fill.
                tag = {
                    "hidden_block_count_x": "HiddenKind::BlockCountX",
                    "hidden_block_count_y": "HiddenKind::BlockCountY",
                    "hidden_block_count_z": "HiddenKind::BlockCountZ",
                    "hidden_group_size_x": "HiddenKind::GroupSizeX",
                    "hidden_group_size_y": "HiddenKind::GroupSizeY",
                    "hidden_group_size_z": "HiddenKind::GroupSizeZ",
                    "hidden_remainder_x": "HiddenKind::RemainderX",
                    "hidden_remainder_y": "HiddenKind::RemainderY",
                    "hidden_remainder_z": "HiddenKind::RemainderZ",
                    "hidden_grid_dims": "HiddenKind::GridDims",
                    "hidden_global_offset_x": "HiddenKind::Zero",
                    "hidden_global_offset_y": "HiddenKind::Zero",
                    "hidden_global_offset_z": "HiddenKind::Zero",
                    "hidden_dynamic_lds_size": "HiddenKind::DynamicLdsSize",
                    "hidden_default_queue": "HiddenKind::Zero",
                    "hidden_completion_action": "HiddenKind::Zero",
                    "hidden_multigrid_sync_arg": "HiddenKind::Zero",
                    "hidden_private_base": "HiddenKind::Zero",
                    "hidden_shared_base": "HiddenKind::Zero",
                    "hidden_queue_ptr": "HiddenKind::Zero",
                    "hidden_heap_v1": "HiddenKind::Zero",
                    "hidden_hostcall_buffer": "HiddenKind::Zero",
                    "hidden_printf_buffer": "HiddenKind::Zero",
                    "hidden_none": "HiddenKind::Zero",
                }.get(kind, "HiddenKind::Zero")
                rs_slots.append(f"        ArgSlot {{ offset: {a['offset']}, size: {a['size']}, kind: ArgKind::Hidden({tag}) }},")
                continue
            else:
                continue
            rs_slots.append(f"        ArgSlot {{ offset: {a['offset']}, size: {a['size']}, kind: {rk} }},")
        slot_const = f"K_{name.upper()}"
        lines.append(f"static {slot_const}: &[ArgSlot] = &[")
        lines.extend(rs_slots)
        lines.append("];")

    # Build the table.
    lines.append("")
    lines.append("pub fn all_layouts() -> HashMap<&'static str, KernelLayout> {")
    lines.append("    let mut m = HashMap::new();")
    for name in sorted(layouts.keys()):
        ksize = kernarg_size_total.get(name, 0)
        slot_const = f"K_{name.upper()}"
        lines.append(f'    m.insert("{name}", KernelLayout {{ kernarg_size: {ksize}, args: {slot_const} }});')
    lines.append("    m")
    lines.append("}")
    lines.append("")

    out_rs.write_text("\n".join(lines))
    print(f"wrote {out_rs} with {len(layouts)} kernels", file=sys.stderr)


if __name__ == "__main__":
    main()
