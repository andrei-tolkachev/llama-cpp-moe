#!/usr/bin/env python3
"""Compile and exercise P07 host Engram row-cache component.

No GGUF/model/GPU/CUDA workload is executed. The module is standalone and NOT
part of a patched llama.cpp executable until the graph adapter is integrated.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd: list[str], timeout: int = 120) -> dict:
    start = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    return {"exit_code": proc.returncode, "stdout": proc.stdout,
            "stderr": proc.stderr, "elapsed_seconds": round(time.monotonic()-start, 3),
            "command": cmd}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--sanitizers", action="store_true")
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    out = args.output.resolve()
    compiler = shutil.which(args.compiler)
    if out.exists() or args.output.is_symlink() or out.is_relative_to(root) or not out.parent.exists():
        parser.error("--output must be a new directory outside the source root")
    if not compiler:
        parser.error("C++ compiler unavailable")
    files = [root / "modules/engram/row_cache.hpp", root / "modules/engram/mapped_rows.hpp",
             root / "tests_cpp/engram_row_cache_fixture.cpp"]
    try:
        for f in files:
            if not f.is_file() or f.is_symlink():
                raise RuntimeError("missing/symlink source file: " + str(f))
        checksums = {str(f.relative_to(root)):digest(f) for f in files}
        out.mkdir(mode=0o700)
        flags = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread"]
        if args.sanitizers:
            flags = ["-std=c++17", "-O1", "-g", "-Wall", "-Wextra", "-Werror", "-pthread",
                     "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-fno-sanitize-recover=all"]
        executable = out / "engram_row_cache_fixture"
        build = run([compiler,*flags,str(files[2]),"-o",str(executable)])
        execution = run([str(executable)], timeout=120) if build["exit_code"] == 0 else None
        success = execution is not None and execution["exit_code"] == 0
        if success:
            try:
                payload = json.loads(execution["stdout"].strip())
                success = payload == {"status":"PASS", "tests":6, "gpu_executed":False,"model_executed":False}
            except (ValueError, TypeError):
                success = False
        status = "PASS" if success else "FAIL"
        report = {"schema_version":1,"status":status,"component":"P07 Engram raw-row cache prototype",
                  "integration_status":"STANDALONE_NOT_ATTACHED_TO_LLAMA_GRAPH",
                  "no_model_or_gpu_tests":True,"sanitizers":args.sanitizers,
                  "input_sha256":checksums,"compiler":run([compiler,"--version"])["stdout"].splitlines()[0],
                  "compile":build,"execute":execution,
                  "output_binary_sha256":digest(executable) if executable.is_file() else None}
        (out / "result.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
        if checksums != {str(f.relative_to(root)):digest(f) for f in files}:
            raise RuntimeError("source mutated during validation")
        print(json.dumps({"status":status, "record":str(out / "result.json"),"gpu_validated":False}))
        return 0 if success else 1
    except (OSError,RuntimeError,subprocess.TimeoutExpired) as error:
        if out.is_dir(): (out / "error.json").write_text(json.dumps({"status":"ERROR","message":str(error)})+"\n")
        print(json.dumps({"status":"ERROR","message":str(error)}))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
