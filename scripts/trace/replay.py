#!/usr/bin/env python3
"""replay.py — render a .trace JSONL file as a readable end-to-end timeline.

The .trace file is produced by iara-engine (IARA_TRACE=layer|byte|full +
IARA_TRACE_FILE=path) and/or byte-language-lab. Each line is one TraceEvent.

Shows, per token (seq), the path through the model: which stage, L2 norm,
top firing neurons, nan flags. Brain + engine events stitch by trace_id.

Usage:
  python3 replay.py <file.trace> [--seq N] [--stage substr] [--full]
"""
import argparse, json, sys
from collections import defaultdict


def load(path):
    evs = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                evs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  warn: line {ln} bad json: {e}", file=sys.stderr)
    return evs


def fmt_topk(top_k, n=4):
    parts = []
    for entry in top_k[:n]:
        idx, val = entry[0], entry[1]
        parts.append(f"#{idx}={val:+.3f}" if isinstance(val, (int, float)) else f"#{idx}={val}")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--seq", type=int, default=None, help="filter to one token index")
    ap.add_argument("--stage", default="", help="filter stage by substring")
    ap.add_argument("--full", action="store_true", help="show full tensor dumps if present")
    args = ap.parse_args()

    evs = load(args.file)
    if not evs:
        print("no events")
        return

    # Group by (trace_id, seq).
    groups = defaultdict(list)
    for e in evs:
        groups[(e.get("tid", 0), e.get("seq", 0))].append(e)

    nan_count = sum(1 for e in evs if e.get("nan"))
    srcs = sorted(set(e.get("src", "?") for e in evs))
    print(f"═══ {args.file} — {len(evs)} events, {len(groups)} (trace,token) groups ═══")
    print(f"    sources: {', '.join(srcs)}   nan events: {nan_count}")
    print()

    for (tid, seq), group in sorted(groups.items()):
        if args.seq is not None and seq != args.seq:
            continue
        print(f"┌─ trace {tid}  token #{seq} ─ {len(group)} stages")
        for e in group:
            stage = e.get("stage", "?")
            if args.stage and args.stage not in stage:
                continue
            l2 = e.get("l2", 0.0)
            nan = " ⚠NAN" if e.get("nan") else ""
            src = e.get("src", "?")[0].upper()
            tk = fmt_topk(e.get("top_k", []))
            l2s = f"{l2:8.3f}" if isinstance(l2, (int, float)) else f"{l2:>8}"
            print(f"│  [{src}] {stage:<22} L2={l2s}  {tk}{nan}")
            if args.full and e.get("full"):
                vals = e["full"][:16]
                print(f"│      full[:16]={vals}")
        print("└─")
        print()


if __name__ == "__main__":
    main()
