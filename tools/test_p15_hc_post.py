#!/usr/bin/env python3
"""Compile host-executed CUDA kernel math against the pinned upstream kernel.

This is not a CUDA, nvcc, model or performance test.  Builds fixture from the
actual R14 candidate modules and an exact upstream function body fixture.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'tests_cpp/hc_post_reference_original.hpp'
FIXTURE=ROOT/'tests_cpp/hc_post_reuse_fixture.cpp'
HEADER=ROOT/'modules/cuda/hc_post_reuse.cuh'

def execute(out: Path, compiler: str, sanitizer: bool)->int:
    binary=out/'hc_post_reuse_host'
    cmd=[compiler,'-std=c++17','-O1','-g','-Wall','-Wextra','-Werror',
         '-Wno-unknown-pragmas','-ffp-contract=off','-fno-fast-math']
    if sanitizer:
        cmd+=['-fsanitize=address,undefined','-fno-sanitize-recover=all',
              '-fno-omit-frame-pointer']
    cmd+=[str(FIXTURE),'-o',str(binary)]
    env=os.environ.copy()
    if sanitizer:
        env['ASAN_OPTIONS']=env.get('ASAN_OPTIONS','')+':detect_leaks=0'
    reports=[]
    for name,args in [('compile',cmd),('run',[str(binary)])]:
        try:
            p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                             text=True,timeout=90,env=env)
            reports.append({'name':name,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
            if p.returncode != 0: break
        except (OSError,subprocess.TimeoutExpired) as e:
            reports.append({'name':name,'error':str(e)})
            break
    ok=len(reports)==2 and all(x.get('returncode')==0 for x in reports)
    result={'status':'PASS_HOST_PARITY_ONLY' if ok else 'FAIL',
            'sanitizers':sanitizer,'reports':reports,
            'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (REF,FIXTURE,HEADER)},
            'not_tested':['NVCC and GPU device','CUDA graph behavior','model logits',
                          'real HC_POST performance']}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if ok else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sanitizers',action='store_true')
    parser.add_argument('--compiler',default='c++')
    parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    if a.output:
        path=a.output.resolve()
        if path.exists() or ROOT==path or ROOT in path.parents:
            parser.error('output directory must not exist and must be outside the source package')
        path.mkdir(parents=True,exist_ok=False)
        return execute(path,a.compiler,a.sanitizers)
    with tempfile.TemporaryDirectory(prefix='llama-p15-') as td:
        return execute(Path(td),a.compiler,a.sanitizers)

if __name__=='__main__':
    raise SystemExit(main())
