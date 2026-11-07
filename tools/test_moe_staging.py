#!/usr/bin/env python3
"""Test actual P04 byte packing with deferred-read and backend-capability fixtures.

The candidate checkout is read only. The generated helper test uses no model,
GPU, CUDA library or service. A helper-test pass is not inference validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_helpers(text: str) -> str:
    begin = "struct moe_cache_slru {"
    end = "// gate, up, down or gate_up, down"
    anchors = (begin, end, "static size_t llama_moe_cache_staging_limit(",
               "static ggml_backend_buffer_ptr llama_moe_cache_alloc_staging(",
               "static moe_cache_upload_stats llama_moe_cache_upload_experts(")
    for anchor in anchors:
        if text.count(anchor) != 1:
            raise ValueError("Candidate source anchor is absent or ambiguous: " + anchor)
    first, last = text.index(begin), text.index(end)
    if last <= first:
        raise ValueError("Unexpected candidate helper ordering")
    # The later P12 wrapper follows the original P04 template.
    wrapper = text.find("// P12: prefer a single immutable group layout", first, last)
    return text[first:wrapper if wrapper >= 0 else last]


def command(argv: list[str], *, env: dict | None = None) -> dict:
    began = time.monotonic()
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=120, check=False, env=env)
    return {"command": argv, "exit_code": result.returncode,
            "elapsed_seconds": round(time.monotonic() - began, 3),
            "stdout": result.stdout, "stderr": result.stderr}


def run(args, output: Path, source: bytes, helpers: str, compiler: str) -> int:
    fixture = Path(__file__).resolve().parents[1] / "tests_cpp/moe_staging_fixture.cpp"
    fixture_bytes = fixture.read_bytes()
    template = fixture_bytes.decode("utf-8")
    anchor = "// @ACTUAL_STAGING_HELPERS@"
    if template.count(anchor) != 1:
        raise ValueError("Fixture helper anchor is absent or ambiguous")
    test_source = output / "moe_staging_generated.cpp"
    test_source.write_text(template.replace(anchor, helpers))
    executable = output / "moe_staging_test"
    version = command([compiler, "--version"])
    flags = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror"]
    compilation = command([compiler, *flags, str(test_source), "-o", str(executable)])
    execution = command([str(executable)]) if compilation["exit_code"] == 0 else None
    normal_pass = execution is not None and execution["exit_code"] == 0
    sanitizer = {"status": "NOT_RUN"}
    if normal_pass and args.sanitizers:
        sanitized = output / "moe_staging_test_sanitized"
        sanitize_flags = ["-std=c++17", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                          "-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]
        sanitize_compile = command([compiler, *sanitize_flags, str(test_source), "-o", str(sanitized)])
        env = dict(os.environ)
        if args.disable_leak_check:
            env["ASAN_OPTIONS"] = "detect_leaks=0"
        sanitize_run = command([str(sanitized)], env=env) if sanitize_compile["exit_code"] == 0 else None
        sanitizer = {
            "status": "PASS" if sanitize_run and sanitize_run["exit_code"] == 0 else "FAIL",
            "compile": sanitize_compile, "execution": sanitize_run,
            "leak_check_explicitly_disabled": args.disable_leak_check,
            "options": {key: env.get(key) for key in ("ASAN_OPTIONS", "LSAN_OPTIONS", "UBSAN_OPTIONS")}}
    passed = normal_pass and sanitizer["status"] != "FAIL"
    report = {
        "schema_version": 1, "status": "PASS" if passed else "FAIL",
        "scope": "Actual production helpers; CPU byte equality and deferred-read lifetime; mock backend capability/fallback. No GPU, inference, linked llama or transfer timing.",
        "candidate_source": str(args.candidate.resolve() / "src/llama-moe-cache.cpp"),
        "candidate_source_sha256": sha256(source),
        "extracted_helpers_sha256": sha256(helpers.encode()),
        "fixture_sha256": sha256(fixture_bytes),
        "test_tool_sha256": sha256(Path(__file__).read_bytes()),
        "compiler_version": version,
        "compile": compilation, "execution": execution, "sanitizers": sanitizer,
        "gpu_executed": False, "inference_executed": False,
        "performance_claim": None}
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate llama source checkout")
    parser.add_argument("--output", type=Path, help="New directory; omitted uses a temporary directory")
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--sanitizers", action="store_true", help="Also run AddressSanitizer and UndefinedBehaviorSanitizer")
    parser.add_argument("--disable-leak-check", action="store_true", help="Explicitly disable LeakSanitizer if the environment prevents /proc thread inspection")
    args = parser.parse_args()
    if args.disable_leak_check and not args.sanitizers:
        parser.error("--disable-leak-check requires --sanitizers")
    try:
        args.candidate = args.candidate.resolve(strict=True)
        if args.output:
            output = args.output.resolve()
            if output.is_relative_to(args.candidate):
                raise ValueError("--output must be outside the candidate source checkout")
            if args.output.exists() or args.output.is_symlink():
                parser.error("--output must not already exist")
        else:
            # Reject configured source-local parents before gettempdir probes
            # them for writability, as well as before TemporaryDirectory.
            for name in ("TMPDIR", "TEMP", "TMP"):
                value = os.environ.get(name)
                if value and Path(value).resolve().is_relative_to(args.candidate):
                    raise ValueError("Temporary output must be outside the candidate source checkout")
            temporary_parent = Path(tempfile.gettempdir()).resolve(strict=True)
            if temporary_parent.is_relative_to(args.candidate):
                raise ValueError("Temporary output must be outside the candidate source checkout")
        compiler = shutil.which(args.compiler)
        if compiler is None:
            parser.error("C++ compiler is unavailable")
        source = (args.candidate / "src/llama-moe-cache.cpp").read_bytes()
        helpers = extract_helpers(source.decode("utf-8"))
        if args.output:
            output.mkdir(parents=False, exist_ok=False)
            return run(args, output, source, helpers, compiler)
        with tempfile.TemporaryDirectory(prefix="llama-moe-staging-", dir=temporary_parent) as temporary:
            return run(args, Path(temporary), source, helpers, compiler)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"status": "ERROR", "message": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
