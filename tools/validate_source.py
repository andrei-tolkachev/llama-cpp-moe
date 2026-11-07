#!/usr/bin/env python3
"""Run source-level acceptance for the cumulative patch package without a GPU.

Compile actual changed/dependent translation units and exercise the real source
helpers with explicit host fixtures. Does not configure CMake, link llama.cpp,
load a GGUF, start a server, install anything, or benchmark inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

UNITS = (
    "src/llama-moe-cache.cpp", "src/llama-context.cpp", "src/llama-graph.cpp", "src/llama-impl.cpp",
    "src/models/deepseek4.cpp", "src/models/deepseek41.cpp", "src/models/qwen4exp.cpp",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root: Path, expected: dict[str, str]) -> dict[str, str]:
    files = {root / name for name in (*UNITS, *expected)}
    for part in ("include", "src", "ggml/include"):
        for pattern in ("*.h", "*.hpp"):
            files.update((root / part).rglob(pattern))
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(files)}


def run_step(command: list[str], logfile: Path, sanitize: bool = False) -> dict:
    start = time.monotonic()
    proc = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=600, shell=False,
                          env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0:halt_on_error=1"}
                          if sanitize else None)
    logfile.write_text(proc.stdout + proc.stderr)
    return {"command": command, "exit_code": proc.returncode,
            "seconds": round(time.monotonic() - start, 3),
            "log": logfile.name, "log_sha256": digest(logfile)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory outside both source trees")
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--sanitize", action="store_true", help="ASan/UBSan helper runs; leak scan disabled in sandbox")
    args = parser.parse_args()
    baseline, candidate = args.baseline.resolve(strict=True), args.candidate.resolve(strict=True)
    output = args.output.resolve()
    if baseline == candidate or any(output.is_relative_to(p) for p in (baseline, candidate)):
        parser.error("Separate baseline/candidate trees and external output are required")
    if output.exists() or args.output.is_symlink() or not output.parent.is_dir():
        parser.error("--output must be a new directory with an existing parent")
    compiler = shutil.which(args.compiler)
    if not compiler:
        parser.error("C++ compiler is unavailable; no output directory created")
    kit = Path(__file__).resolve().parents[1]
    manifest = json.loads((kit / "manifest.json").read_text())
    base_expected, candidate_expected = {}, {}
    for patch in manifest["patches"]:
        for target in patch["targets"]:
            base_expected.setdefault(target["path"], target["before_sha256"])
            candidate_expected[target["path"]] = target["after_sha256"]
    for asset in manifest["additional_files"]:
        if digest(kit / asset["source"]) != asset["sha256"]:
            parser.error("Package header checksum mismatch: " + asset["source"])
        candidate_expected[asset["destination"]] = asset["sha256"]
    trees = (("baseline", baseline, base_expected), ("candidate", candidate, candidate_expected))
    try:
        before = {role: snapshot(tree, expected) for role, tree, expected in trees}
    except OSError as exc:
        parser.error(str(exc))
    for role, _, expected in trees:
        for name, checksum in expected.items():
            if before[role][name] != checksum:
                parser.error(f"{role} differs from the pinned manifest at {name}")
    output.mkdir(mode=0o700, exist_ok=False)
    report = {"schema_version": 1, "status": "RUNNING", "source_files": before,
              "release_id": manifest["release_id"], "base_commit": manifest["upstream"]["base_commit"],
              "scope": "Source/host-only acceptance; no full engine link, CUDA, model inference or speed validation",
              "compiler": subprocess.run([compiler, "--version"], capture_output=True, text=True, check=True).stdout.splitlines()[0],
              "steps": [], "not_run": ["full CMake build and linking", "CUDA execution", "numerical model parity",
                                        "KV/long-context/cancel runtime", "images/video/tools API runtime", "tok/s measurements"]}
    record = output / "source_validation.json"
    def save():
        record.write_text(json.dumps(report, indent=2) + "\n")
    save()
    code = 0
    try:
        for role, tree in (("baseline", baseline), ("candidate", candidate)):
            for name in UNITS:
                label = role + "_" + Path(name).stem
                print("Compiling " + role + " " + name, file=sys.stderr, flush=True)
                cmd = [compiler, "-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror", "-c",
                       *["-I" + str(tree / p) for p in ("include", "src", "src/models", "ggml/include")],
                       str(tree / name), "-o", str(output / (label + ".o"))]
                step = run_step(cmd, output / (label + ".log"))
                report["steps"].append(step); save()
                if step["exit_code"]:
                    raise RuntimeError("Translation-unit compilation failed: " + label)
                step["object_sha256"] = digest(output / (label + ".o"))
        tools = Path(__file__).resolve().parent
        cases = [
            ("moe_plan", ["--sanitizers", "--disable-leak-check"]),
            ("moe_staging", ["--sanitizers", "--disable-leak-check"]),
            ("hc_post", ["--sanitize"]),
            ("hash_scratch", ["--sanitize"]),
            ("prefetch_rows", ["--sanitize"]),
        ]
        for name, sanitizer_args in cases:
            print("Checking " + name, file=sys.stderr, flush=True)
            cmd = [sys.executable, str(tools / ("test_" + name + ".py")),
                   "--candidate", str(candidate), "--output", str(output / name), "--compiler", compiler]
            if name in ("hash_scratch", "prefetch_rows"):
                cmd += ["--baseline", str(baseline)]
            if args.sanitize:
                cmd += sanitizer_args
            step = run_step(cmd, output / (name + ".log"), args.sanitize)
            report["steps"].append(step); save()
            if step["exit_code"]:
                raise RuntimeError("Host-source check failed: " + name)
        # The final P14B bridge covers the P04/P09/P10/P12 call sites, scheduler
        # and event lifetime together against the installed candidate headers.
        # Historical bridges extract earlier layouts which are no longer present.
        for name in (
                "engram_cache", "engram_stage", "engram_model_cache",
                "moe_pipeline", "moe_worker", "moe_metadata", "moe_fill_plan",
                "moe_scheduler", "moe_upload_completion", "moe_scheduler_bridge",
                "p15_hc_post", "p15b_mmq_tile", "p15c_mmq_probe",
                "p16a_mtp_window", "p16a_source_bridge", "p16b_mtp_guard"):
            print("Checking " + name, file=sys.stderr, flush=True)
            cmd = [sys.executable, str(tools / ("test_" + name + ".py")),
                   "--output", str(output / name), "--compiler", compiler]
            if name == "moe_scheduler_bridge":
                cmd += ["--candidate", str(candidate)]
            if args.sanitize:
                cmd.append("--sanitizers")
            step = run_step(cmd, output / (name + ".log"), args.sanitize)
            report["steps"].append(step)
            save()
            if step["exit_code"]:
                raise RuntimeError("Host-source check failed: " + name)
        after = {role: snapshot(tree, expected) for role, tree, expected in trees}
        if before != after:
            raise RuntimeError("Inspected source/header files changed")
        report["source_files_unchanged"] = True
        report["status"] = "PASS_SOURCE_HOST_ONLY"
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        code = 1; report.update(status="FAIL", error=str(error))
    finally:
        save()
    print(json.dumps({"status": report["status"], "record": str(record), "full_engine_or_gpu_validated": False}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
