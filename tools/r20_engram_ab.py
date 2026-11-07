#!/usr/bin/env python3
"""Measure exact staged raw rows in memory. Requires an unpacked, original R19 kit."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tempfile


def run():
    here = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r19-kit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="new result JSON outside package")
    p.add_argument("--cpu", type=int, help="pin test process to one available CPU")
    args = p.parse_args()
    if args.out.exists():
        p.error("--out must be new")
    source = here / "tests_cpp/r20_engram_measure.cpp"
    compiler = shutil.which("g++")
    if not compiler:
        p.error("g++ required")
    env = dict(os.environ, LLAMA_EXPERIMENTAL_ENGRAM_TRACE="0")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    flags = ["-std=c++17", "-O3", "-DNDEBUG", "-Wall", "-Wextra", "-Werror", "-pthread"]
    record = dict(scope="CPU-only, resident byte array; no mmap faults, CUDA or model inference measured",
                  compiler=subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
                  host=platform.platform(), cpu_affinity=sorted(os.sched_getaffinity(0)), flags=flags,
                  cpp_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), inputs={}, runs=[], summaries={})
    model = Path("/proc/cpuinfo")
    if model.is_file():
        record["cpu_model"] = next((line.split(":", 1)[1].strip() for line in model.read_text().splitlines()
                                   if line.startswith("model name")), None)
    with tempfile.TemporaryDirectory() as temp:
        exe = {}
        for label, kit in [("R19", args.r19_kit), ("R20", here)]:
            headers = kit / "modules" / "engram"
            record["inputs"][label] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                       for f in sorted(headers.glob("*.hpp"))}
            exe[label] = str(Path(temp) / label)
            subprocess.run([compiler, *flags, "-I", str(kit / "modules"), str(source), "-o", exe[label]], check=True)
        for case in ["duplicate", "warm", "cold"]:
            def sample(label, warmup, block):
                proc = subprocess.run([exe[label], case], capture_output=True, text=True, timeout=30,
                                      env=env, check=True)
                data = json.loads(proc.stdout)
                record["runs"].append(dict(case=case, label=label, warmup=warmup, block=block,
                                           stdout=proc.stdout.strip(), stderr=proc.stderr, **data))
                if case == "duplicate":
                    expected = data["rows"] * (2 if label == "R19" else 1)
                    if data["reader_calls"] != expected:
                        raise RuntimeError("duplicate-read regression")
            for label in exe:
                sample(label, True, None)
            for block in range(6):
                order = ["R19", "R20", "R20", "R19"] if block % 2 == 0 else ["R20", "R19", "R19", "R20"]
                for label in order:
                    sample(label, False, block)
            results = [x for x in record["runs"] if x["case"] == case and not x["warmup"]]
            if case == "duplicate":
                continue
            if len({(x["digest"], x["source_reads"], x["hits"]) for x in results}) != 1:
                raise RuntimeError("inconsistent row bytes or hits/misses")
            for label in exe:
                series = [r["wall_ns"] / r["rows"] for r in results if r["label"] == label]
                record["summaries"][case + "_" + label] = dict(
                    n=len(series), median_ns_per_row=statistics.median(series),
                    p95_nearest_rank_ns_per_row=sorted(series)[math.ceil(len(series) * .95) - 1],
                    sample_variance_ns2_per_row2=statistics.variance(series))
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(json.dumps(record["summaries"], indent=2))


if __name__ == "__main__":
    run()
