#!/usr/bin/env python3
"""Compile P12 shared fill plan with exact P04 patch oracle; host-only."""
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

ROOT=Path(__file__).resolve().parents[1]

def sha(data): return hashlib.sha256(data).hexdigest()

def original_helper():
    p=(ROOT/'patches/0004-moe-upload-staging.patch').read_text()
    start=p.index('+struct moe_cache_upload_stats {')
    end=p.index(' // gate, up, down or gate_up, down',start)
    lines=p[start:end].splitlines(keepends=True)
    assert all(l.startswith('+') for l in lines), 'P04 helper has unexpected diff content'
    return ''.join(l[1:] for l in lines)

def invoke(args,env=None):
    start=time.monotonic()
    r=subprocess.run(args,capture_output=True,text=True,env=env,timeout=90)
    return {'status':'PASS' if r.returncode==0 else 'FAIL','code':r.returncode,
            'elapsed_s':round(time.monotonic()-start,3),
            'stdout':r.stdout[-2500:],'stderr':r.stderr[-5000:]}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler',default='c++')
    p.add_argument('--sanitizers',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    c=shutil.which(a.compiler)
    if not c: p.error('compiler not found')
    if a.output and (a.output.exists() or a.output.resolve().is_relative_to(ROOT)):
        p.error('output must not exist and must be outside source')
    with tempfile.TemporaryDirectory(prefix='llama-p12-') as tmp:
        out=Path(tmp) if a.output is None else a.output.resolve()
        out.mkdir(exist_ok=True)
        template=(ROOT/'tests_cpp/moe_fill_plan_fixture.cpp.in').read_text()
        assert template.count('@@P04_HELPER@@')==1
        content=template.replace('@@P04_HELPER@@',original_helper())
        source=out/'moe_fill_plan_fixture.cpp';source.write_text(content)
        steps=[]
        for mode in ('plain', 'sanitized'):
            if mode=='sanitized' and not a.sanitizers: continue
            exe=out/f'moe_fill_{mode}'
            command=[c,'-std=c++17','-pthread','-Wall','-Wextra','-Werror','-I'+str(ROOT/'modules'),
                     '-O2' if mode=='plain' else '-O1',str(source),'-o',str(exe)]
            if mode=='sanitized':
                command += ['-g','-fno-omit-frame-pointer','-fsanitize=address,undefined','-fno-sanitize-recover=all']
            step={'mode':mode,'compile':invoke(command)}
            if step['compile']['code']==0:
                for flag in ('enabled','disabled'):
                    env=os.environ.copy()
                    if flag=='enabled':env['LLAMA_EXPERIMENTAL_MOE_SHARED_PLAN']='1'
                    else:env.pop('LLAMA_EXPERIMENTAL_MOE_SHARED_PLAN',None)
                    if mode=='sanitized':
                        env['ASAN_OPTIONS']='detect_leaks=0:halt_on_error=1'
                        env['UBSAN_OPTIONS']='halt_on_error=1'
                    step[flag]=invoke([str(exe),flag],env)
            steps.append(step)
            if step['compile']['code'] or any(step.get(flag,{}).get('code',1) for flag in ('enabled','disabled')):break
        passed=len(steps)==(2 if a.sanitizers else 1) and all(
            x['compile']['code']==0 and x['enabled']['code']==0 and x['disabled']['code']==0 for x in steps)
        report={'schema':'llama.experimental.p12.host.v1','status':'PASS' if passed else 'FAIL',
                'scope':'host-only P04 exact semantics, thread sharing and ASan/UBSan; no CUDA or full llama.cpp build',
                'p04_patch_sha256':sha((ROOT/'patches/0004-moe-upload-staging.patch').read_bytes()),
                'p12_header_sha256':sha((ROOT/'modules/moe/upload_plan.hpp').read_bytes()),
                'generated_sha256':sha(source.read_bytes()),'steps':steps}
        if a.output:(out/'result.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(report,indent=2,ensure_ascii=False))
        return 0 if passed else 1

if __name__=='__main__':sys.exit(main())
