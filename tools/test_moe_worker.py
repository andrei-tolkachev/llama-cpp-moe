#!/usr/bin/env python3
"""P10 host-only: compile persistent worker and original P09B caller path.

Never starts llama.cpp, CUDA, GPUs, servers, or modifies source trees.
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

ROOT = Path(__file__).resolve().parents[1]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def call(cmd, env=None):
    t = time.monotonic()
    p = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                       text=True, env=env, timeout=75)
    return {'cmd':cmd,'exit_code':p.returncode,'seconds':round(time.monotonic()-t,3),
            'stdout':p.stdout,'stderr':p.stderr}


def run(source: Path, output: Path, compiler: str, sanitizers: bool) -> int:
    tracked = ['modules/moe/upload_types.hpp', 'modules/moe/upload_worker.hpp',
               'modules/moe/upload_pipeline.hpp',
               'tests_cpp/moe_upload_worker_fixture.cpp',
               'tests_cpp/moe_upload_pipeline_fixture.cpp']
    before = {k:sha(source/k) for k in tracked}
    steps = []
    for fixture in ('moe_upload_worker_fixture.cpp', 'moe_upload_pipeline_fixture.cpp'):
        for sanitized in ([False, True] if sanitizers else [False]):
            exe = output/(fixture.removesuffix('.cpp')+('_san' if sanitized else '_plain'))
            cmd=[compiler,'-std=c++17','-Wall','-Wextra','-Werror','-pedantic',
                 '-pthread','-I'+str(source),'-I'+str(source/'modules'),
                 str(source/'tests_cpp'/fixture),'-o',str(exe)]
            if sanitized:
                cmd += ['-O1','-g','-fno-omit-frame-pointer',
                        '-fsanitize=address,undefined','-fno-sanitize-recover=all']
            else: cmd += ['-O2']
            steps.append(call(cmd))
            if steps[-1]['exit_code']: break
            # Check compatibility of the same P09B call-site in both modes.
            for mode in ('old', 'persistent'):
                env=dict(os.environ)
                if mode=='persistent':env['LLAMA_EXPERIMENTAL_MOE_PERSISTENT']='1'
                else:env.pop('LLAMA_EXPERIMENTAL_MOE_PERSISTENT',None)
                if sanitized:
                    env['ASAN_OPTIONS']='detect_leaks=0:halt_on_error=1'
                    env['UBSAN_OPTIONS']='halt_on_error=1'
                steps.append(call([str(exe)],env))
                if steps[-1]['exit_code']: break
            if any(x['exit_code'] for x in steps):break
        if any(x['exit_code'] for x in steps):break
    after = {k:sha(source/k) for k in tracked}
    ok=(before==after and all(x['exit_code']==0 for x in steps) and
        len(steps)==(12 if sanitizers else 6))
    result={'schema':'llama.experimental.p10.persistent_host.v1',
            'status':'PASS' if ok else 'FAIL','sanitizers':sanitizers,
            'source_files_unchanged':before==after,'hashes':before,'steps':steps,
            'scope':'C++17 host only; no full llama.cpp TU/CUDA, no runtime/gpu-speed claims',
            'not_run':['real PCIe DMA','long context model parity','GPU profiling']}
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'checks':len(steps)}))
    return 0 if ok else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--compiler',default='c++')
    parser.add_argument('--sanitizers',action='store_true')
    args=parser.parse_args()
    try:
        source=args.source_root.resolve(strict=True)
        compiler=shutil.which(args.compiler)
        if not compiler: raise ValueError('compiler missing')
        if args.output is not None:
            dest=args.output.resolve()
            if dest.exists() or dest.is_symlink() or dest.is_relative_to(source):
                raise ValueError('output must not exist and must be outside source root')
            if not dest.parent.is_dir(): raise ValueError('output parent does not exist')
            dest.mkdir(mode=0o700)
            return run(source,dest,compiler,args.sanitizers)
        with tempfile.TemporaryDirectory(prefix='llama-exp-p10-') as d:
            return run(source,Path(d),compiler,args.sanitizers)
    except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
        print(json.dumps({'status':'ERROR','reason':str(exc)}))
        return 2

if __name__=='__main__':sys.exit(main())
