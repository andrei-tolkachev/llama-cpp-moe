#!/usr/bin/env python3
"""Compile actual P03 HC_POST policy snippets against instrumented host mocks.

Requires a patched llama.cpp source tree, its real headers, and a C++17 compiler.
No CMake, CUDA, model loading, inference, or network operation is performed.
This is a control-flow contract test, not numerical or paired GPU validation.
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


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def between(text: str, start: str, end: str) -> str:
    """Fail clearly when a forward port no longer has the reviewed boundaries."""
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("Source anchor is absent or ambiguous: " + start)
    first, last = text.index(start), text.index(end)
    if last <= first:
        raise ValueError("Source anchors are out of order: " + start)
    return text[first:last]


def run(args: argparse.Namespace, output: Path) -> int:
    candidate = args.candidate
    template = Path(__file__).resolve().parents[1] / "tests_cpp/hc_post_policy_fixture.cpp.in"
    paths = (
        "src/models/deepseek4.cpp", "src/llama-context.cpp", "src/llama-graph.h",
        "src/llama-cparams.h", "include/llama.h", "ggml/include/ggml.h",
        "ggml/include/ggml-backend.h", "ggml/include/ggml-alloc.h",
        "ggml/include/ggml-cpu.h", "ggml/include/ggml-opt.h", "ggml/include/gguf.h",
    )
    source_bytes = {path: (candidate / path).read_bytes() for path in paths}
    model = source_bytes["src/models/deepseek4.cpp"].decode("utf-8")
    context = source_bytes["src/llama-context.cpp"].decode("utf-8")
    graph = source_bytes["src/llama-graph.h"].decode("utf-8")
    parts = {
        "LAYOUT_AND_BUILDER": between(model, "static bool dsv4_hc_post_layer_layout(",
                                       "ggml_tensor * llama_model_deepseek4::graph::build_hc_head("),
        "CONTEXT_POLICY": between(context, "        const char * hc_post_layers_env = getenv(",
                                  "        if (cparams.moe_cache_size > 0) {"),
        "REUSE_POLICY": between(graph, "        if (cparams.dsv4_hc_post_backends !=",
                                "        return\n            cparams.embeddings"),
    }
    template_bytes = template.read_bytes()
    body = template_bytes.decode("utf-8")
    for name, part in parts.items():
        marker = "@" + name + "@"
        if body.count(marker) != 1:
            raise ValueError("Fixture marker is absent or ambiguous: " + name)
        body = body.replace(marker, part)
    generated = output / "hc_post_policy_generated.cpp"
    generated.write_text(body, encoding="utf-8")
    binary = output / "hc_post_policy_test"
    command = [args.compiler, "-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror"]
    if args.sanitize:
        command += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer"]
    command += ["-I" + str(candidate / folder) for folder in ("include", "src", "ggml/include")]
    command += [str(generated), "-o", str(binary)]
    env = os.environ.copy()
    if args.sanitize:
        # LeakSanitizer cannot enumerate /proc tasks in the development sandbox.
        # Address/UB checks remain enabled; the report excludes leak coverage.
        previous = env.get("ASAN_OPTIONS", "")
        env["ASAN_OPTIONS"] = (previous + ":" if previous else "") + "detect_leaks=0"
    records = []
    status = 0
    for kind, cmd in (("host_fixture_compile", command), ("host_control_flow_test", [str(binary)])):
        start = time.monotonic()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
            record = dict(kind=kind, command=cmd, exit_code=result.returncode,
                          stdout=result.stdout, stderr=result.stderr)
            status = result.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            record = dict(kind=kind, command=cmd, exit_code=None, error=str(error))
            status = 1
        record["elapsed_seconds"] = round(time.monotonic() - start, 3)
        records.append(record)
        if status:
            break
    unchanged = all((candidate / path).read_bytes() == data for path, data in source_bytes.items())
    if not unchanged:
        status = 1
    report = dict(
        schema_version=1,
        status="PASS_HOST_CONTROL_FLOW_ONLY" if status == 0 else "FAIL",
        scope="Actual candidate policy/builder/reuse source; real headers; instrumented GGML/backend/scheduler mocks",
        candidate=str(candidate),
        source_header_sha256={path: sha256(data) for path, data in source_bytes.items()},
        source_snippet_sha256={name: sha256(part.encode("utf-8")) for name, part in parts.items()},
        template_sha256=sha256(template_bytes),
        generated_source_sha256=sha256(generated.read_bytes()),
        inspected_source_files_unchanged=unchanged,
        sanitizer="AddressSanitizer + UndefinedBehaviorSanitizer; leak detection disabled" if args.sanitize else "none",
        records=records,
        not_validated=["full build or linking of llama.cpp", "real scheduler placement", "CUDA support or execution",
                       "floating-point equivalence", "leak detection", "model inference", "speedup"],
    )
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if status == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True, help="Patched source checkout with real upstream headers")
    parser.add_argument("--output", type=Path, help="New directory for generated host fixture, binary and result.json")
    parser.add_argument("--compiler", default="c++", help="C++ compiler executable, without shell arguments")
    parser.add_argument("--sanitize", action="store_true", help="Enable ASan and UBSan; leak checks remain disabled")
    args = parser.parse_args()
    try:
        args.candidate = args.candidate.resolve(strict=True)
        if not args.candidate.is_dir():
            raise ValueError("--candidate must be a source directory")
        if args.output:
            output = args.output.resolve()
            if output == args.candidate or args.candidate in output.parents:
                raise ValueError("--output must be outside the candidate source tree")
            output.mkdir(parents=False, exist_ok=False)
            return run(args, output)
        with tempfile.TemporaryDirectory(prefix="llama-hc-post-") as temporary:
            return run(args, Path(temporary))
    except (OSError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
