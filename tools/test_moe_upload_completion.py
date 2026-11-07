#!/usr/bin/env python3
"""P14B host-only completion event lifetime and fail-closed tests.

No CUDA device is used: the fake backend delays each H2D source read until
backend/event synchronize; the real ggml integration requires native build.
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

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(args, env=None):
    start=time.monotonic()
    try:
        p=subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,text=True,timeout=40,env=env)
        return {'rc':p.returncode,'seconds':round(time.monotonic()-start,3),
                'out':p.stdout[-700:],'err':p.stderr[-900:]}
    except subprocess.TimeoutExpired:
        return {'rc':124,'seconds':round(time.monotonic()-start,3),'err':'TIMEOUT'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--compiler',default='c++')
    p.add_argument('--sanitizers',action='store_true')
    args=p.parse_args()
    compiler=shutil.which(args.compiler)
    if not compiler:p.error('C++ compiler unavailable')
    if args.output and (args.output.exists() or args.output.resolve().is_relative_to(ROOT)):
        p.error('output must be a new directory outside source')
    src=ROOT/'tests_cpp/moe_upload_completion_fixture.cpp'
    if not src.is_file() or not (ROOT/'modules/moe/upload_completion.hpp').is_file():
        p.error('P14B source files missing')
    with tempfile.TemporaryDirectory(prefix='llama-p14b-') as td:
        output=args.output.resolve() if args.output else Path(td)
        if args.output:output.mkdir(mode=0o700)
        report={'schema':'llama.experimental.p14b.host.v1','status':'FAIL',
          'scope':'host-only fake backend events + deferred DMA; not CUDA compile/runtime/performance',
          'fixture_sha256':sha(src),'header_sha256':sha(ROOT/'modules/moe/upload_completion.hpp'),
          'runs':[]}
        for mode in (['plain','sanitizers'] if args.sanitizers else ['plain']):
            exe=output/('completion_'+mode)
            cmd=[compiler,'-std=c++17','-Wall','-Wextra','-Werror','-pthread',
                 '-O1' if mode=='sanitizers' else '-O2',str(src),'-o',str(exe)]
            if mode=='sanitizers':cmd+=['-g','-fno-omit-frame-pointer',
                         '-fsanitize=address,undefined','-fno-sanitize-recover=all']
            step={'mode':mode,'compile':run(cmd),'cases':[]}
            if step['compile']['rc']==0:
                for flag in ('', '0', '1', 'true', '11'):
                    env=dict(os.environ)
                    if flag:env['LLAMA_EXPERIMENTAL_MOE_EVENT_FENCE']=flag
                    else:env.pop('LLAMA_EXPERIMENTAL_MOE_EVENT_FENCE',None)
                    if mode=='sanitizers':
                        env['ASAN_OPTIONS']='detect_leaks=0:halt_on_error=1'
                        env['UBSAN_OPTIONS']='halt_on_error=1'
                    step['cases'].append({'flag':flag or '<unset>',**run([str(exe)],env)})
                if mode=='plain':
                    check=run([str(exe),'--unclosed'],env)
                    step['fail_closed_unreleased_event']={'rc':check['rc'],
                        'expected_nonzero':True,'passed':check['rc'] not in (0,124)}
            report['runs'].append(step)
            if step['compile']['rc'] or any(c['rc'] for c in step['cases']):break
        report['status']='PASS' if len(report['runs'])==(2 if args.sanitizers else 1) and all(
            x['compile']['rc']==0 and len(x['cases'])==5 and
            all(y['rc']==0 for y in x['cases']) and
            (x['mode']!='plain' or x['fail_closed_unreleased_event']['passed'])
            for x in report['runs']) else 'FAIL'
        if args.output:(output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'modes':[
          {'mode':s['mode'],'compile':s['compile']['rc'],'runs':[c['rc'] for c in s['cases']]}
          for s in report['runs']]}))
        return 0 if report['status']=='PASS' else 1

if __name__=='__main__':sys.exit(main())
