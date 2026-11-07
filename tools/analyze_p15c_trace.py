#!/usr/bin/env python3
"""Analyze opt-in P15C CUDA MMQ *host selector* stderr JSONL.

No CUDA commands or subprocesses. Results are not GPU timings or tok/s.
Example: python3 tools/analyze_p15c_trace.py run.log
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PREFIX = "P15C_MMQ_SELECT_JSON "
REQUIRED = {"thread", "type", "moe", "ncols_bucket", "calls", "cache_hits", "select_ns", "max_select_ns"}


def checked_int(value: Any, name: str, low: int = 0) -> int:
    if type(value) is not int or value < low:
        raise ValueError(f"{name}: expected integer >= {low}")
    return value


def accumulate(lines) -> dict:
    by_key: dict[tuple[int, bool, int], dict] = {}
    count, overflow, threads = 0, 0, set()
    for line_no, line in enumerate(lines, 1):
        if not line.startswith(PREFIX):
            continue
        if len(line) > 4096:
            raise ValueError(f"line {line_no}: oversized profile record")
        try:
            item = json.loads(line[len(PREFIX):])
        except json.JSONDecodeError as err:
            raise ValueError(f"line {line_no}: invalid profile JSON: {err}") from err
        if not isinstance(item, dict):
            raise ValueError(f"line {line_no}: expected a JSON object")
        thread = checked_int(item.get("thread"), "thread", 1)
        threads.add(thread)
        if "overflow" in item:
            if set(item) != {"thread", "overflow"}:
                raise ValueError(f"line {line_no}: malformed overflow record")
            overflow += checked_int(item["overflow"], "overflow", 1)
            continue
        if set(item) != REQUIRED or type(item["moe"]) is not bool:
            raise ValueError(f"line {line_no}: unsupported profile fields")
        type_id = checked_int(item["type"], "type")
        bucket = checked_int(item["ncols_bucket"], "ncols_bucket")
        if bucket > 5:
            raise ValueError(f"line {line_no}: unknown column bucket")
        calls = checked_int(item["calls"], "calls", 1)
        hits = checked_int(item["cache_hits"], "cache_hits")
        elapsed = checked_int(item["select_ns"], "select_ns")
        max_elapsed = checked_int(item["max_select_ns"], "max_select_ns")
        if hits > calls or max_elapsed > elapsed:
            raise ValueError(f"line {line_no}: internally inconsistent counters")
        key = (type_id, item["moe"], bucket)
        row = by_key.setdefault(key, {"type": type_id, "moe": item["moe"],
                       "ncols_bucket": bucket, "calls": 0, "cache_hits": 0,
                       "select_ns": 0, "max_select_ns": 0})
        row["calls"] += calls
        row["cache_hits"] += hits
        row["select_ns"] += elapsed
        row["max_select_ns"] = max(row["max_select_ns"], max_elapsed)
        count += 1
    if count == 0:
        raise ValueError("no valid P15C profile records found")
    groups = []
    for row in by_key.values():
        c = row["calls"]
        groups.append({**row, "avg_select_ns": round(row["select_ns"]/c, 2),
                       "cache_hit_rate": round(row["cache_hits"]/c, 6)})
    groups.sort(key=lambda x: (-x["select_ns"], -x["calls"], x["type"], x["ncols_bucket"]))
    total_calls = sum(x["calls"] for x in groups)
    total_ns = sum(x["select_ns"] for x in groups)
    hits = sum(x["cache_hits"] for x in groups)
    return {"kind":"HOST_TILE_SELECTOR_ONLY", "records":count,
            "threads":len(threads), "dropped_samples":overflow,
            "total_calls":total_calls, "total_cache_hits":hits,
            "host_selection_time_ms":round(total_ns/1e6, 6),
            "avg_select_ns_per_call":round(total_ns/total_calls, 2),
            "groups":groups,
            "warnings":(["Overflow occurred: rankings are incomplete"] if overflow else []) +
              ["This is CPU-side tile-selection instrumentation, not GPU kernel time or tok/s"]}


def read_logs(paths: list[Path]) -> dict:
    def source():
        for path in paths:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                yield from f
    return accumulate(source())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+", help="Captured stderr log(s) from P15C profile")
    parser.add_argument("--output", type=Path, help="Write summary JSON to a new file (no overwrite)")
    args = parser.parse_args()
    try:
        report = read_logs(args.logs)
        serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as f:
                f.write(serialized)
        else:
            sys.stdout.write(serialized)
        return 0
    except (ValueError, OSError) as err:
        parser.error(str(err))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
