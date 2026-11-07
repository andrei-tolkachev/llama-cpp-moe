#!/usr/bin/env python3
"""Validate EXACT candidate P04 helper + P09B backend submission block.

Host-only: the fake backend defers device reads until synchronize; no CUDA,
no full llama.cpp compilation, and no model/speed/parity claims.
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

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(argv, env=None):
    t0=time.monotonic()
    p=subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,text=True,
                     timeout=75,check=False,env=env)
    return {"argv": argv,"exit_code":p.returncode,"seconds":round(time.monotonic()-t0,3),
            "stdout":p.stdout,"stderr":p.stderr}

def extract(src: str, start: str, stop: str) -> str:
    assert src.count(start) == 1, f"missing or repeated source marker: {start}"
    a=src.index(start)
    b=src.find(stop,a+len(start))
    assert b>a, f"missing end marker: {stop}"
    return src[a:b]

def build(candidate: Path, output: Path, compiler: str, sanitizers:bool) -> int:
    target=candidate/'src/llama-moe-cache.cpp'
    expected=ROOT/'modules/moe/upload_pipeline.hpp'
    vendored=candidate/'src/llama-experimental/upload_pipeline.hpp'
    if not target.is_file() or not vendored.is_file():
        raise ValueError("candidate missing patched source or vendored upload_pipeline.hpp")
    if sha(vendored)!=sha(expected):
        raise ValueError("candidate header differs from release header")
    plan_expected=ROOT/'modules/moe/upload_plan.hpp'
    plan_vendor=candidate/'src/llama-experimental/upload_plan.hpp'
    if not plan_vendor.is_file() or sha(plan_vendor)!=sha(plan_expected):
        raise ValueError("candidate P12 shared fill plan header missing or differs from release header")
    manifest=json.loads((ROOT/'manifest.json').read_text())
    p12=next(x for x in manifest['patches'] if x['id']=='P12-moe-shared-fill-plan')
    if sha(target)!=p12['targets'][0]['after_sha256']:
        raise ValueError("Expected the P12 historical target. For the current candidate use test_moe_scheduler_bridge.py.")
    src=target.read_text()
    helper=extract(src, 'struct moe_cache_upload_stats {','// gate, up, down or gate_up, down')
    snippet=extract(src,'        std::optional<llama_exp::moe_fill_plan> shared_plan;',
                         '        stats & st = ids.size()')
    assert snippet.count('moe_upload_pipeline::execute(')==1
    assert snippet.count('llama_moe_cache_upload_experts_planned(')==2
    assert snippet.count('ggml_backend_tensor_set_async')>=2
    return run_fixture(candidate, output, compiler, sanitizers, helper, snippet, sha(target),
                       "P04, P09B and P12 literal source with a deferred backend")

def run_fixture(candidate, output, compiler, sanitizers, helper, snippet, source_digest, scope):
    tmpl=(ROOT/'tests_cpp/moe_upload_bridge_fixture.cpp.in').read_text()
    code=tmpl.replace('@@P04_HELPER@@',helper).replace('@@P09B_COPY_BLOCK@@',snippet)
    assert '@@' not in code
    generated=output/'moe_bridge_generated.cpp'
    generated.write_text(code)
    command=[compiler,'-std=c++17','-Wall','-Wextra','-Werror','-pthread',
             '-I'+str(candidate/'src'),str(generated)]
    steps=[]
    obj=output/'moe_bridge_host'
    steps.append(run([*command,'-O2','-o',str(obj)]))
    for mode in ('enabled','disabled'):
        for shared in ('shared','baseline'):
            if steps[-1]['exit_code']: break
            env=dict(os.environ)
            if mode=='enabled':env['LLAMA_EXPERIMENTAL_MOE_PIPELINE']='1'
            else:env.pop('LLAMA_EXPERIMENTAL_MOE_PIPELINE',None)
            if shared=='shared':env['LLAMA_EXPERIMENTAL_MOE_SHARED_PLAN']='1'
            else:env.pop('LLAMA_EXPERIMENTAL_MOE_SHARED_PLAN',None)
            steps.append(run([str(obj),mode,shared],env=env))
    if sanitizers and all(p['exit_code']==0 for p in steps):
        bin=output/'moe_bridge_sanitized'
        steps.append(run([*command,'-O1','-g','-fno-omit-frame-pointer',
                    '-fsanitize=address,undefined','-fno-sanitize-recover=all','-o',str(bin)]))
        for mode in ('enabled','disabled'):
            for shared in ('shared','baseline'):
                if steps[-1]['exit_code']: break
                env=dict(os.environ)
                env['ASAN_OPTIONS']='detect_leaks=0:halt_on_error=1'
                env['UBSAN_OPTIONS']='halt_on_error=1'
                if mode=='enabled':env['LLAMA_EXPERIMENTAL_MOE_PIPELINE']='1'
                else:env.pop('LLAMA_EXPERIMENTAL_MOE_PIPELINE',None)
                if shared=='shared':env['LLAMA_EXPERIMENTAL_MOE_SHARED_PLAN']='1'
                else:env.pop('LLAMA_EXPERIMENTAL_MOE_SHARED_PLAN',None)
                steps.append(run([str(bin),mode,shared],env=env))
    passed=len(steps)==(10 if sanitizers else 5) and all(x['exit_code']==0 for x in steps)
    report={'schema':'llama.experimental.p12.mock_backend.v1',
            'status':'PASS' if passed else 'FAIL',
            'checks':scope,
            'original_patched_source_sha256':source_digest,
            'generated_fixture_sha256':sha(generated),
            'sanitizers':sanitizers,
            'scope':'host-only: no actual ggml backend / GPU / full CMake / CUDA / GGUF inference',
            'steps':steps}
    (output/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':report['status'],'report':str(output/'result.json')}))
    return 0 if passed else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--compiler',default='c++')
    parser.add_argument('--sanitizers',action='store_true')
    a=parser.parse_args()
    try:
        candidate=a.candidate.resolve(strict=True)
        compiler=shutil.which(a.compiler)
        if not compiler:raise ValueError('C++ compiler not found')
        if a.output:
            o=a.output.resolve()
            if o.exists() or o.is_relative_to(candidate) or o.is_relative_to(ROOT):
                raise ValueError('output must not exist and cannot be inside source or candidate')
            if not o.parent.is_dir():raise ValueError('output parent missing')
            o.mkdir(mode=0o700)
            return build(candidate,o,compiler,a.sanitizers)
        with tempfile.TemporaryDirectory(prefix='llama-exp-p09b-') as td:
            return build(candidate,Path(td),compiler,a.sanitizers)
    except (OSError,ValueError,AssertionError,subprocess.TimeoutExpired) as exc:
        print(json.dumps({'status':'ERROR','reason':str(exc)}))
        return 2

if __name__=='__main__':sys.exit(main())
