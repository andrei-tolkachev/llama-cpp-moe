#!/usr/bin/env python3
"""Compile and compare actual graph-input methods using host boundary fixtures.

This is not a complete llama build, GPU test, Engram storage benchmark, or
numerical test of model logits. Full production translation units are checked
separately; here backend writes and prefetch calls are captured as exact bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def function(text: str, start: str) -> str:
    if text.count(start) != 1:
        raise ValueError("Source anchor is absent or ambiguous: " + start)
    a = text.index(start)
    opening = text.index("{", a)
    depth = 0
    for i in range(opening, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[a:i+1]
    raise ValueError("Unbalanced source function: " + start)


def graph_class(text: str, name: str) -> str:
    return function(text, "class " + name + " :") + ";"


def section(text: str, kind: str, optimized: bool) -> str:
    if kind == "qwen":
        name = "llm_graph_input_qwen4exp_ple"
        helpers = ([function(text, "static bool qwen4exp_hash_scratch_enabled()"),
                    function(text, "static void qwen4exp_hash_rows_reuse(")]
                   if optimized else [])
        methods = [function(text, "void " + name + "::set_input(")]
    else:
        name = "llm_graph_input_dsv41"
        helpers = [function(text, "static bool dsv41_hash_scratch_enabled()")] if optimized else []
        methods = [function(text, "void " + name + "::set_input("),
                   function(text, "bool " + name + "::can_reuse(")]
    return "\n\n".join([*helpers, graph_class(text, name), *methods])


def run(args, output: Path) -> int:
    template = Path(__file__).resolve().parents[1] / "tests_cpp/hash_scratch_fixture.cpp"
    body = template.read_text()
    sources = {}
    for role, tree in (("baseline", args.baseline), ("candidate", args.candidate)):
        for kind, filename in (("qwen", "qwen4exp.cpp"), ("deepseek", "deepseek41.cpp")):
            p = tree / "src/models" / filename
            text = p.read_text()
            sources[role + ":" + filename] = hashlib.sha256(p.read_bytes()).hexdigest()
            body = body.replace("// @" + role.upper() + "_" + kind.upper() + "@",
                                section(text, kind, role == "candidate"))
    base_hash = (args.baseline / "src/models/deepseek41.cpp").read_text()
    body = body.replace("// @ACTUAL_ENGRAM_HASH@",
                        function(base_hash, "void llama_model_deepseek41::engram_hash("))
    source = output / "hash_scratch_generated.cpp"
    source.write_text(body)
    executable = output / "hash_scratch_test"
    command = [args.compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
               "-Wno-mismatched-new-delete", "-pthread", "-I" + str(args.candidate / "src"), str(source), "-o", str(executable)]
    if args.sanitize:
        command[1:1] = ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]
    began = time.monotonic()
    compile_run = subprocess.run(command, capture_output=True, text=True, timeout=120)
    result = {"scope": "Actual extracted graph-input methods; fixture tensor backend and KV history; no GPU/inference",
              "source_sha256": sources, "command": command,
              "compile_exit_code": compile_run.returncode, "compile_stderr": compile_run.stderr}
    if compile_run.returncode == 0:
        env = os.environ.copy()
        if args.sanitize:
            # The execution sandbox does not permit LeakSanitizer /proc task scans.
            env["ASAN_OPTIONS"] = "detect_leaks=0"
            result["sanitizers"] = {"address": True, "undefined": True, "leaks": False}
        test = subprocess.run([str(executable)], capture_output=True, text=True, timeout=120, env=env)
        result.update(test_exit_code=test.returncode, stdout=test.stdout, stderr=test.stderr)
        status = test.returncode
    else:
        status = compile_run.returncode
    result["elapsed_seconds"] = round(time.monotonic() - began, 3)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New local directory for generated test and result")
    parser.add_argument("--compiler", default="c++")
    parser.add_argument("--sanitize", action="store_true", help="ASan/UBSan; leak scanning disabled for sandbox")
    args = parser.parse_args()
    args.baseline = args.baseline.resolve(strict=True)
    args.candidate = args.candidate.resolve(strict=True)
    if not args.baseline.is_dir() or not args.candidate.is_dir():
        parser.error("Both inputs must be source directories")
    if args.output:
        args.output = args.output.resolve()
        if any(args.output.is_relative_to(p) for p in (args.baseline, args.candidate)):
            parser.error("--output must be outside both source trees")
        args.output.mkdir(parents=False, exist_ok=False)
        return run(args, args.output)
    with tempfile.TemporaryDirectory(prefix="llama-hash-scratch-") as temporary:
        return run(args, Path(temporary))


if __name__ == "__main__":
    raise SystemExit(main())
