#!/usr/bin/env python3
"""Linux cold/warm A/B of exact pinned llama_prefetch on private GGUF-shaped expert ranges."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tmp-parent", type=Path, default=Path.cwd(), help="Filesystem under test; creates one private 192-MiB file")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    source = (args.source / "src/llama-mmap.cpp").read_text()
    kernel = source[source.index("void llama_prefetch(llama_memory_ranges mr) {"):source.index("size_t llama_path_max()")]
    route_source = (args.source / "src/llama-moe-cache.cpp").read_text()
    route = route_source[route_source.index("        if (l.readahead && !batched && !fills.empty()) {"):
                         route_source.index("        // The EXISTING readback synchronize above")]
    with tempfile.TemporaryDirectory(dir=args.tmp_parent) as tmp:
        tmp = Path(tmp).resolve()
        cpp, exe = tmp/"bench.cpp", tmp/"probe"
        cpp.write_text((ROOT/"tests_cpp/r20_strata_read.cpp.in").read_text().replace("@@PREFETCH@@", kernel).replace("@@ROUTE@@", route))
        subprocess.run(["c++", "-O2", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I"+str((args.source/"src").resolve()), str(cpp), "-o", str(exe)], check=True)
        file = tmp/"private.weights"
        block = os.urandom(1048576)
        with file.open("xb") as f:
            for i in range(192):
                f.write(i.to_bytes(4, "little")+block[4:])
            f.flush()
            os.fsync(f.fileno())
        report = dict(scope="local Linux mapped file, no inference or GPU", platform=platform.platform(),
                      source_sha256=hashlib.sha256(cpp.read_bytes()).hexdigest(),
                      route_source_sha256=hashlib.sha256(route_source.encode()).hexdigest(), sample_file_bytes=file.stat().st_size,
                      fs=subprocess.check_output(["stat", "-f", "-c", "%T", str(tmp)],text=True).strip(),
                      expert_width_bytes=1048576, experts_per_bank=64, banks=3,
                      routed_experts=[2,11,22,37,45,59], schedule="ABBA BAAB ABBA BAAB", rejected_preconditions=[], runs=[], summary={})
        # Wide-separated source experts map to the same requested bytes in both arms.
        for policy in ("normal", "random"):
            for state in ("cold","warm"):
                key = policy+"_"+state
                output_hash = None
                for arm in "ABBABAABABBABAAB":
                    for attempt in range(12):
                        row = json.loads(subprocess.check_output([str(exe),str(file),arm,state,policy],text=True))
                        if row["resident_before"] == (row["pages"] if state == "warm" else 0):
                            break
                        report["rejected_preconditions"].append(dict(policy=policy, state=state, arm=arm, **row))
                    else:
                        raise RuntimeError(f"Could not set initial cache state: {policy} {state} {row}")
                    if output_hash is not None and row["checksum"] != output_hash:
                        raise ValueError("byte parity failure")
                    output_hash = row["checksum"]
                    if row["resident_before"] != (row["pages"] if state == "warm" else 0):
                        raise ValueError(f"cache precondition not established: {row}")
                    report["runs"].append(dict(policy=policy, state=state, arm=arm,**row))
                summary = report["summary"][key] = {}
                for arm in "AB":
                    sub = [r for r in report["runs"] if r["policy"]==policy and r["state"]==state and r["arm"]==arm]
                    t = sorted(r["total_ms"] for r in sub)
                    summary[arm] = dict(n=len(t),median_ms=statistics.median(t),p95_ms=t[math.ceil(.95*len(t))-1],
                                        variance_ms2=statistics.variance(t),median_major_faults=statistics.median(r["major"] for r in sub),
                                        median_read_bytes=statistics.median(r["read_bytes"] for r in sub))
        with args.out.open("x") as out:
            json.dump(report,out,indent=2)
            out.write("\n")
        print(json.dumps(report["summary"]))


if __name__=="__main__":
    main()
