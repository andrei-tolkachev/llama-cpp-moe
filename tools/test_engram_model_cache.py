#!/usr/bin/env python3
"""P13 deterministic C++17 host parity/unload tests; never a GPU benchmark."""
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

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    'tests_cpp/engram_model_cache_fixture.cpp',
    'modules/engram/graph_stage.hpp',
    'modules/engram/model_cache.hpp',
    'modules/engram/row_cache.hpp',
    'modules/engram/mapped_rows.hpp',
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str], timeout: int, env: dict[str,str] | None = None) -> dict:
    start = time.monotonic()
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=timeout, env=env, check=False)
        return {'code': p.returncode, 'stdout': p.stdout[-4000:],
                'stderr': p.stderr[-4000:], 'seconds': round(time.monotonic()-start, 3)}
    except subprocess.TimeoutExpired:
        return {'code': -1, 'error': 'timeout', 'seconds': timeout}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, default=ROOT)
    ap.add_argument('--compiler', default='c++')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--sanitizers', action='store_true')
    args = ap.parse_args()
    source = args.source_root.resolve(strict=True)
    out = args.output.resolve()
    cc = shutil.which(args.compiler)
    if (not cc or out.exists() or args.output.is_symlink() or out.is_relative_to(source)
            or not out.parent.is_dir()
            or any(not (source/p).is_file() or (source/p).is_symlink() for p in FILES)):
        ap.error('compiler and regular input files are required; output must be new, outside source')
    before = {p: sha(source/p) for p in FILES}
    out.mkdir(mode=0o700)
    flags = ['-std=c++17', '-Wall', '-Wextra', '-Werror', '-pthread']
    if args.sanitizers:
        flags += ['-O1', '-g', '-fno-omit-frame-pointer', '-fsanitize=address,undefined',
                  '-fno-sanitize-recover=all']
    else:
        flags += ['-O2']
    binary = out/'p13_host'
    compile_result = run([cc, *flags, str(source/FILES[0]), '-o', str(binary)], 100)
    execute_result: dict = {'code': -1, 'error': 'compile failed'}
    if compile_result['code'] == 0:
        env=os.environ.copy()
        if args.sanitizers:
            env['ASAN_OPTIONS']='detect_leaks=0:halt_on_error=1'
            env['UBSAN_OPTIONS']='halt_on_error=1'
        execute_result = run([str(binary)], 100, env)
    try: result = json.loads(execute_result.get('stdout', ''))
    except (json.JSONDecodeError, TypeError): result = None
    passed = (compile_result['code'] == 0 and execute_result['code'] == 0 and
              result == {'status':'PASS','scenarios':4,'gpu':False,'model':False} and
              before == {p: sha(source/p) for p in FILES})
    data = {
        'schema': 'llama.experimental.p13.host.v1', 'status': 'PASS' if passed else 'FAIL',
        'sanitizers': args.sanitizers, 'source_sha256': before,
        'compile': compile_result, 'execute': execute_result,
        'model_or_cuda_runtime_executed': False,
        'full_llama_cpp_translation_unit_compiled': False,
    }
    (out/'result.json').write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps({'status': data['status'], 'result': str(out/'result.json')}))
    return 0 if passed else 1

if __name__ == '__main__':
    raise SystemExit(main())
