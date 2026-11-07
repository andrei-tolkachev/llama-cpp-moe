#!/usr/bin/env python3
"""Compile and exercise the P09 host-only producer/consumer scheduling module.

Does not link a llama.cpp binary, invoke a CUDA GPU, or measure PCIe overlap.
The fixture emulates deferred async backend reads to check lifetime and order.
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
import tempfile
import time


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(command: list[str], *, env=None, timeout=120) -> dict:
    start = time.monotonic()
    proc = subprocess.run(command, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, env=env, timeout=timeout, check=False)
    return {"exit_code": proc.returncode, "command": command,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "stdout": proc.stdout, "stderr": proc.stderr}


def run(source: Path, output: Path, compiler: str, sanitizers: bool) -> int:
    hdr = source / 'modules/moe/upload_pipeline.hpp'
    fixture = source / 'tests_cpp/moe_upload_pipeline_fixture.cpp'
    if not hdr.is_file() or not fixture.is_file():
        raise ValueError('P09 source files are missing')
    # Host-only, independent of the source tree, never edits the release.
    common = [compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pthread',
              '-I'+str(source / 'modules'), str(fixture)]
    standard_binary = output / 'moe_upload_pipeline_host'
    steps = [execute([*common, '-O2', '-o', str(standard_binary)])]
    if steps[-1]['exit_code'] == 0:
        steps.append(execute([str(standard_binary)]))
    if sanitizers and all(s['exit_code'] == 0 for s in steps):
        sanitized_binary = output / 'moe_upload_pipeline_sanitized'
        steps.append(execute([*common, '-O1', '-g', '-fsanitize=address,undefined',
                              '-fno-sanitize-recover=all', '-fno-omit-frame-pointer',
                              '-o', str(sanitized_binary)]))
        if steps[-1]['exit_code'] == 0:
            env = dict(os.environ)
            env['ASAN_OPTIONS'] = 'detect_leaks=0:halt_on_error=1'
            env['UBSAN_OPTIONS'] = 'halt_on_error=1'
            steps.append(execute([str(sanitized_binary)], env=env))
    passed = all(s['exit_code'] == 0 for s in steps)
    if len(steps) != (4 if sanitizers else 2):
        passed = False
    result = {
        'schema': 'llama.experimental.p09.host.v1',
        'status': 'PASS' if passed else 'FAIL',
        'source_module': 'P09 host upload producer, NOT integrated into llama-moe-cache.cpp',
        'header_sha256': sha(hdr), 'fixture_sha256': sha(fixture),
        'compiler': compiler, 'sanitizers': sanitizers,
        'asan_ubsan': 'PASS' if sanitizers and passed else ('NOT_RUN' if not sanitizers else 'FAIL'),
        'leaks_checked': False,
        'steps': steps,
        'not_tested': ['full llama.cpp build/link', 'CUDA', 'GPU stream concurrency',
                       'real pinned-memory status', 'PCIe transfer bandwidth',
                       'model inference', 'performance or numerical parity'],
    }
    (output / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': result['status'],
                      'report': str(output / 'result.json'),
                      'full_gpu_validated': False}))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, help='nonexistent directory outside source root')
    parser.add_argument('--compiler', default='c++')
    parser.add_argument('--sanitizers', action='store_true')
    args = parser.parse_args()
    try:
        source = args.source_root.resolve(strict=True)
        compiler = shutil.which(args.compiler)
        if not compiler:
            raise ValueError('C++17 compiler is unavailable')
        if args.output:
            dest = args.output.resolve()
            if dest.exists() or dest.is_symlink() or dest.is_relative_to(source):
                raise ValueError('output must not exist and must be outside source tree')
            if not dest.parent.is_dir():
                raise ValueError('output parent does not exist')
            dest.mkdir(mode=0o700)
            return run(source, dest, compiler, args.sanitizers)
        with tempfile.TemporaryDirectory(prefix='llama-experimental-p09-') as temp:
            return run(source, Path(temp), compiler, args.sanitizers)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(json.dumps({'status': 'ERROR', 'reason': str(error)}))
        return 2


if __name__ == '__main__':
    sys.exit(main())
