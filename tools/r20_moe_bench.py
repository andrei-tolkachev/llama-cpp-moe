#!/usr/bin/env python3
"""Time the candidate's exact upload block on a deferred host memory backend."""
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

from test_moe_scheduler_bridge import ROOT, extract


def build(candidate, directory):
    source = (candidate / "src/llama-moe-cache.cpp").read_text()
    bridge = (ROOT / "tests_cpp/moe_scheduler_bridge_fixture.cpp.in").read_text()
    code = "#include <chrono>\n" + bridge[:bridge.index("static void run_case(")]
    for marker, start, stop in (
        ("@@P04_HELPER@@", "struct moe_cache_upload_stats {", "// gate, up, down or gate_up, down"),
        ("@@P09B_COPY_BLOCK@@", "        std::optional<llama_exp::moe_fill_plan> shared_plan;", "        stats & st = ids.size()"),
        ("@@RECOVERY@@", "        if (!g.upload_complete) {", "        size_t n_hit = 0;"),
        ("@@COMMIT@@", "        g.upload_complete = true;", "        return true;"),
        ("@@P14B_RECORD_BLOCK@@", "        // P14B: event is queued AFTER the final slot-map write", "        if (trace.enabled) {"),
    ):
        code = code.replace(marker, extract(source, start, stop))
    code += (ROOT / "tests_cpp/r20_moe_bench_main.cpp.in").read_text()
    cpp = directory / "bench.cpp"
    exe = directory / "moe-bench"
    cpp.write_text(code)
    subprocess.run(["c++", "-std=c++17", "-O2", "-pthread", "-Wall", "-Wextra", "-Werror",
                    "-I" + str(candidate / "src"), str(cpp), "-o", str(exe)], check=True)
    return exe, hashlib.sha256(source.encode()).hexdigest(), hashlib.sha256(code.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1024)
    parser.add_argument("--r19-control", action="store_true",
                        help="Initial diagnosis on unchanged R19: compare pipeline vs serial P04")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.updates < 100:
        parser.error("at least 100 updates required")
    with tempfile.TemporaryDirectory() as tmp:
        exe, source_hash, code_hash = build(args.candidate.resolve(), Path(tmp))
        report = dict(scope="CPU mock backend, warm memory, actual extracted submit block; no model or CUDA",
                      phase="R19 control" if args.r19_control else "R20 no-pack admission toggle",
                      candidate_source_sha256=source_hash, generated_source_sha256=code_hash,
                      cpu=next((s.partition(":")[2].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                                if s.startswith("model name")), platform.machine()),
                      affinity=sorted(os.sched_getaffinity(0)), host=platform.platform(),
                      compiler=subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
                      warmup_updates=64, expert_width_bytes=4096, banks=3, runs=[], summary={})
        # Three crossed blocks. Different styles share workload shape/data and measured updates.
        for shape in ("one", "scattered", "direct", "packable"):
            for arm in "ABBABAABABBA":
                env = {k:v for k,v in os.environ.items() if not k.startswith("LLAMA_EXPERIMENTAL_")}
                env["LLAMA_EXPERIMENTAL_MOE_PIPELINE"] = str(int(not args.r19_control or arm == "A"))
                env["LLAMA_EXPERIMENTAL_MOE_BYPASS_NO_PACK"] = str(int(not args.r19_control and arm == "B"))
                record = json.loads(subprocess.check_output([str(exe), shape, str(args.updates)], text=True, env=env))
                expect_pipeline = not args.r19_control or arm == "A"
                expect_inline = not args.r19_control and arm == "B" and shape != "packable"
                if (record["pipeline_updates"] != (args.updates if expect_pipeline else 0) or
                        record["inline_updates"] != (args.updates if expect_inline else 0)):
                    raise ValueError(f"Unexpected admission for {arm} {shape}")
                report["runs"].append(dict(arm=arm, **record))
            summary = report["summary"][shape] = {}
            for arm in "AB":
                values = sorted(r["us_per_update"] for r in report["runs"] if r["shape"] == shape and r["arm"] == arm)
                summary[arm] = dict(n=len(values), median_us=statistics.median(values),
                                    sample_variance_us2=statistics.variance(values), p95_us=values[math.ceil(.95*len(values))-1])
            summary["B_minus_A_us"] = summary["B"]["median_us"] - summary["A"]["median_us"]
        with args.output.open("x") as out:
            json.dump(report,out,indent=2)
            out.write("\n")
        print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
