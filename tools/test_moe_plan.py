#!/usr/bin/env python3
"""Test the real legacy/optimized SLRU planners extracted from --candidate.

The input may be a complete isolated llama.cpp checkout or a partial source
tree containing src/llama-moe-cache.cpp. A C++17 compiler is sufficient. This
does not build or run llama, CUDA, models, or inference. The optional benchmark
measures this CPU-local planner only; it does not measure token throughput.
The candidate source is read only. All generated files live in a new output
directory, or a newly created temporary directory when --output is omitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def planner(text):
    start = text.index("struct moe_cache_slru {")
    end = text.index("\n};", start) + len("\n};")
    return text[start:end] + "\n"


def method(text, name):
    start = text.index("    bool " + name + "(")
    end = text.index("\n    }\n", start) + len("\n    }\n")
    return text[start:end]


def command(argv, output, *, affinity=None, env=None):
    before_exec = None
    if affinity is not None:
        before_exec = lambda: os.sched_setaffinity(0, {affinity})
    proc = subprocess.run(argv, cwd=output, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=300, check=False, shell=False,
                          preexec_fn=before_exec, env=env)
    if proc.returncode:
        raise RuntimeError("Command failed: " + repr(argv) + "\n" + proc.stdout + proc.stderr)
    return proc.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New directory outside the source tree (default: new temporary directory)")
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--bench", action="store_true", help="Also run the optional CPU-only microbenchmark")
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--case", action="append", default=[], help="Benchmark case name; repeat to select multiple cases")
    parser.add_argument("--pin-cpu", type=int, help="Pin only the owned test process to one permitted CPU")
    parser.add_argument("--sanitizers", action="store_true", help="Run AddressSanitizer and UndefinedBehaviorSanitizer")
    parser.add_argument("--disable-leak-check", action="store_true", help="Explicitly disable only LeakSanitizer when sandbox /proc access prevents it")
    args = parser.parse_args()
    candidate = args.candidate.expanduser().resolve()
    source = candidate / "src/llama-moe-cache.cpp"
    if not candidate.is_dir() or not source.is_file() or source.is_symlink():
        parser.error("candidate must contain a regular src/llama-moe-cache.cpp")
    if args.iterations <= 0 or args.repeats <= 0:
        parser.error("iterations/repeats must be positive")
    if args.case and not args.bench:
        parser.error("--case requires --bench")
    if args.disable_leak_check and not args.sanitizers:
        parser.error("--disable-leak-check requires --sanitizers")
    if args.pin_cpu is not None and (not hasattr(os, "sched_getaffinity") or args.pin_cpu not in os.sched_getaffinity(0)):
        parser.error("requested CPU is unavailable in the current affinity mask")
    compiler = shutil.which(args.compiler)
    if compiler is None:
        parser.error("compiler is unavailable")
    compiler = str(Path(compiler).resolve())
    source_bytes = source.read_bytes()
    text = source_bytes.decode("utf-8")
    extracted = planner(text)
    fallback = method(text, "plan")
    optimized_method = method(text, "plan_reuse")
    harness = Path(__file__).resolve().parents[1] / "tests_cpp/test_moe_plan.cpp"
    harness_bytes = harness.read_bytes()
    if args.output is None:
        temporary_parent = Path(tempfile.gettempdir()).resolve()
        if temporary_parent.is_relative_to(candidate):
            parser.error("default temporary parent is inside candidate; choose --output outside it")
        output = Path(tempfile.mkdtemp(prefix="llama-moe-plan-tests-", dir=temporary_parent)).resolve()
    else:
        requested = args.output.expanduser().absolute()
        if requested.exists() or requested.is_symlink() or not requested.parent.is_dir():
            parser.error("output must be new, with an existing parent")
        output = requested.parent.resolve() / requested.name
        if output.is_relative_to(candidate) or candidate.is_relative_to(output):
            parser.error("output must be outside the candidate source tree")
        output.mkdir(mode=0o700, parents=False, exist_ok=False)
    if output.is_relative_to(candidate):
        raise RuntimeError("temporary directory is inside candidate; choose --output outside it")
    header = output / "planner_under_test.h"
    header.write_text(extracted)
    test_source = output / "test_moe_plan.cpp"
    test_source.write_bytes(harness_bytes)
    report = {"status": "RUNNING", "candidate": str(candidate), "source": str(source),
              "source_sha256": sha(source_bytes), "extracted_planner_sha256": sha(header.read_bytes()),
              "fallback_plan_sha256": sha(fallback.encode()), "optimized_plan_sha256": sha(optimized_method.encode()),
              "test_harness_sha256": sha(harness_bytes), "inference_executed": False, "gpu_executed": False,
              "performance_scope": "CPU-local planner only; not inference, token throughput, or target-PC evidence",
              "commands": []}
    try:
        report["compiler"] = command([compiler, "--version"], output).splitlines()[0]
        binary = output / "test_moe_plan"
        compile_argv = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-O3",
                        "-I", str(output), str(test_source), "-o", str(binary)]
        report["commands"].append(compile_argv)
        command(compile_argv, output)
        run_argv = [str(binary)]
        if args.bench:
            run_argv += ["--bench", str(args.iterations), str(args.repeats)]
            if args.case:
                run_argv.append(",".join(args.case))
        report["commands"].append(run_argv)
        raw = command(run_argv, output, affinity=args.pin_cpu)
        (output / "optimized_stdout.jsonl").write_text(raw)
        samples = [json.loads(line) for line in raw.splitlines() if line]
        report["raw_samples"] = samples
        report["differential"] = [s for s in samples if s["type"] == "differential"]
        if not report["differential"] or any(s["status"] != "PASS" for s in report["differential"]):
            raise RuntimeError("differential test result missing or failed")
        measured_names = {s["case"] for s in samples if s["type"] == "benchmark"}
        if not set(args.case) <= measured_names:
            raise RuntimeError("one or more requested benchmark cases were not measured")
        report["benchmark_summaries"] = []
        for name in sorted(measured_names):
            rows = [s for s in samples if s.get("case") == name]
            ratios = [s["speedup"] for s in rows]
            report["benchmark_summaries"].append({
                "case": name, "samples": len(rows), "iterations_per_sample": rows[0]["iterations"],
                "baseline_median_ns_per_plan": statistics.median(s["baseline_ns_per_plan"] for s in rows),
                "reuse_median_ns_per_plan": statistics.median(s["reuse_ns_per_plan"] for s in rows),
                "paired_speedup_median": statistics.median(ratios), "paired_speedup_min": min(ratios),
                "paired_speedup_max": max(ratios),
                "hit_rate_unique_requests": rows[0]["hits"]/(rows[0]["hits"] + rows[0]["misses"]),
            })
        report["sanitizers"] = {"status": "NOT_RUN"}
        if args.sanitizers:
            binary = output / "test_moe_plan_sanitized"
            sanitize_argv = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-O1", "-g",
                             "-fno-omit-frame-pointer", "-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-I", str(output),
                             str(test_source), "-o", str(binary)]
            report["commands"].append(sanitize_argv)
            command(sanitize_argv, output)
            sanitizer_env = None
            if args.disable_leak_check:
                sanitizer_env = dict(os.environ)
                sanitizer_env["ASAN_OPTIONS"] = "detect_leaks=0"
            raw = command([str(binary)], output, affinity=args.pin_cpu, env=sanitizer_env)
            (output / "sanitized_stdout.jsonl").write_text(raw)
            report["sanitizers"] = {"status": "PASS", "results": [json.loads(line) for line in raw.splitlines() if line],
                                    "leak_detection": "DISABLED_EXPLICITLY" if args.disable_leak_check else "ENABLED"}
        report["source_unchanged"] = source.read_bytes() == source_bytes
        if not report["source_unchanged"]:
            raise RuntimeError("candidate source changed during validation")
        report["status"] = "PASS"
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        report["status"] = "FAIL"
        report["error"] = str(error)
    cpuinfo = Path("/proc/cpuinfo")
    try:
        models = sorted({line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                         if line.startswith("model name")}) if cpuinfo.exists() else []
    except OSError:
        models = []
    report["hardware"] = {"machine": platform.machine(), "cpu_models": models, "pinned_cpu": args.pin_cpu}
    result = output / "results.json"
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "results": str(result), "source_sha256": report["source_sha256"],
                      "differential": report.get("differential"), "sanitizers": report.get("sanitizers", {}).get("status"),
                      "benchmark_summaries": report.get("benchmark_summaries", []), "error": report.get("error")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
